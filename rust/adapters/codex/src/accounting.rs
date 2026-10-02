#[cfg(test)]
use super::paths::collect_codex_usage_files;
use super::{
    loader::dedupe_codex_events,
    parser::{CodexAccountingObservation, visit_codex_session_file_with_diagnostics},
    paths::{CodexUsageFileGroup, codex_usage_sources, collect_deduped_codex_usage_files},
    replay::CodexReplayPlan,
};
use crate::{
    CodexTokenUsageEvent, PricingMap, Result, aggregate_events, calculate_codex_model_cost,
    cli::{AgentReportKind, CodexSpeed, SharedArgs},
    cli_error, filter_events_by_date, print_json_or_jq, resolve_codex_speed,
};
use serde::Serialize;
use serde_json::{Value, json};
#[cfg(test)]
use std::path::Path;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Usage {
    input_tokens: u64,
    cached_input_tokens: u64,
    cache_creation_tokens: u64,
    output_tokens: u64,
    reasoning_output_tokens: u64,
    total_tokens: u64,
}

impl Usage {
    fn add(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_tokens += other.cache_creation_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_output_tokens += other.reasoning_output_tokens;
        self.total_tokens += other.total_tokens;
    }
    fn event(event: &CodexTokenUsageEvent) -> Self {
        Self {
            input_tokens: event.input_tokens,
            cached_input_tokens: event.cached_input_tokens,
            cache_creation_tokens: event.cache_creation_tokens,
            output_tokens: event.output_tokens,
            reasoning_output_tokens: event.reasoning_output_tokens,
            total_tokens: event.total_tokens,
        }
    }
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Coverage {
    observations: u64,
    counter_tokens: u64,
    timestamps_missing: u64,
}

#[derive(Default)]
struct Thread {
    canonical: bool,
    parent: Option<String>,
    fork: Option<String>,
    history_base: Option<String>,
    hierarchy_ambiguous: bool,
    source_sessions: BTreeSet<String>,
    coverage: BTreeMap<String, Coverage>,
}

#[derive(Clone, Default)]
struct Unit {
    usage: Usage,
    priced_tokens: u64,
    unpriced_tokens: u64,
    known_cost: f64,
    requests: u64,
    models: BTreeMap<String, Model>,
}

#[derive(Clone, Default)]
struct Model {
    usage: Usage,
    priced_tokens: u64,
    unpriced_tokens: u64,
    known_cost: f64,
}

impl Unit {
    fn add(&mut self, other: &Self) {
        self.usage.add(&other.usage);
        self.priced_tokens += other.priced_tokens;
        self.unpriced_tokens += other.unpriced_tokens;
        self.known_cost += other.known_cost;
        self.requests += other.requests;
        for (name, model) in &other.models {
            let target = self.models.entry(name.clone()).or_default();
            target.usage.add(&model.usage);
            target.priced_tokens += model.priced_tokens;
            target.unpriced_tokens += model.unpriced_tokens;
            target.known_cost += model.known_cost;
        }
    }
    fn json(&self) -> Value {
        json!({"usage":self.usage,"requestEvents":self.requests,"pricedTokens":self.priced_tokens,"unpricedTokens":self.unpriced_tokens,"knownCostUSD":self.known_cost,"costUSD":(self.unpriced_tokens==0).then_some(self.known_cost),"costPerMillionPricedTokens":(self.priced_tokens>0).then(|| self.known_cost*1_000_000.0/self.priced_tokens as f64),"models":self.models.iter().map(|(name, model)| json!({"model":(!name.is_empty()).then_some(name),"usage":model.usage,"pricedTokens":model.priced_tokens,"unpricedTokens":model.unpriced_tokens,"knownCostUSD":model.known_cost,"costUSD":(model.unpriced_tokens==0).then_some(model.known_cost)})).collect::<Vec<_>>()})
    }
}

pub(crate) fn run(
    shared: &SharedArgs,
    speed: CodexSpeed,
    workstream_map: Option<&str>,
) -> Result<()> {
    let pricing =
        PricingMap::load_with_overrides(shared.offline, false, shared.pricing_overrides.iter());
    let groups = collect_deduped_codex_usage_files(&codex_usage_sources()?);
    let report = report_from_files(&groups, shared, speed, &pricing, workstream_map)?;
    print_json_or_jq(report, shared.jq.as_deref(), shared.no_cost)
}

#[cfg(test)]
fn report_from_directory(
    directory: &Path,
    shared: &SharedArgs,
    speed: CodexSpeed,
    pricing: &PricingMap,
    workstream_map: Option<&str>,
) -> Result<Value> {
    let groups = vec![CodexUsageFileGroup {
        dir: directory.to_path_buf(),
        files: collect_codex_usage_files(directory),
    }];
    report_from_files(&groups, shared, speed, pricing, workstream_map)
}

fn root_thread(id: &str, threads: &BTreeMap<String, Thread>) -> (String, bool) {
    let mut current = id.to_owned();
    let mut seen = BTreeSet::new();
    while seen.insert(current.clone()) {
        let Some(thread) = threads.get(&current) else {
            return (current, false);
        };
        if thread.hierarchy_ambiguous {
            return (current, true);
        };
        let Some(parent) = &thread.parent else {
            return (current, false);
        };
        current = parent.clone();
    }
    (id.to_owned(), true)
}

fn read_workstream_map(path: Option<&str>) -> Result<BTreeMap<String, String>> {
    let Some(path) = path else {
        return Ok(BTreeMap::new());
    };
    let mapping: BTreeMap<String, String> = serde_json::from_slice(&fs::read(path)?)?;
    if mapping
        .iter()
        .any(|(id, label)| id.trim().is_empty() || label.trim().is_empty())
    {
        return Err(cli_error(
            "Workstream mappings require nonempty thread IDs and labels",
        ));
    }
    Ok(mapping)
}

fn report_from_files(
    groups: &[CodexUsageFileGroup],
    shared: &SharedArgs,
    speed: CodexSpeed,
    pricing: &PricingMap,
    workstream_path: Option<&str>,
) -> Result<Value> {
    let workstreams = read_workstream_map(workstream_path)?;
    let replay = CodexReplayPlan::new(
        groups
            .iter()
            .map(|group| (group.dir.as_path(), group.files.as_slice())),
        shared.single_thread,
    );
    let mut threads = BTreeMap::<String, Thread>::new();
    let mut events = Vec::new();
    for group in groups {
        for file in &group.files {
            let mut file_events = Vec::new();
            let mut observations = Vec::new();
            visit_codex_session_file_with_diagnostics(
                &group.dir,
                file,
                replay.replay_prefix(file),
                |event| {
                    file_events.push(event);
                    Ok(())
                },
                |observation| observations.push(observation),
            )?;
            let metadata = observations
                .iter()
                .find_map(|observation| match observation {
                    CodexAccountingObservation::CanonicalMetadata {
                        source_session_id,
                        thread_id,
                        parent_thread_id,
                        forked_from_id,
                        history_base_thread_id,
                    } => Some((
                        source_session_id.clone(),
                        thread_id.clone(),
                        parent_thread_id.clone(),
                        forked_from_id.clone(),
                        history_base_thread_id.clone(),
                    )),
                    _ => None,
                });
            let (source, id, parent, fork, history) = metadata
                .unwrap_or_else(|| (file.to_string_lossy().into_owned(), None, None, None, None));
            let key = id.clone().unwrap_or_else(|| format!("file:{source}"));
            let thread = threads.entry(key.clone()).or_default();
            if !thread.source_sessions.is_empty() && thread.parent != parent {
                thread.hierarchy_ambiguous = true;
            }
            thread.canonical = id.is_some();
            thread.parent = parent;
            thread.fork = fork;
            thread.history_base = history;
            thread.source_sessions.insert(source);
            let mut unresolved_events = Vec::new();
            for observation in observations {
                if let CodexAccountingObservation::UsageCoverage {
                    source_session_id,
                    timestamp,
                    kind,
                    tokens,
                } = observation
                {
                    if kind == "unresolved_initial_counter_without_last"
                        || kind == "unresolved_counter_reset_without_last"
                    {
                        unresolved_events.push(timestamp.clone());
                    }
                    thread.source_sessions.insert(source_session_id);
                    let coverage = thread.coverage.entry(kind).or_default();
                    coverage.observations += 1;
                    coverage.counter_tokens += tokens.map_or(0, |usage| usage.total_tokens);
                    coverage.timestamps_missing += u64::from(timestamp.is_none());
                }
            }
            for mut event in file_events {
                if let Some(index) = unresolved_events
                    .iter()
                    .position(|timestamp| timestamp.as_deref() == Some(event.timestamp.as_str()))
                {
                    unresolved_events.remove(index);
                    let coverage = thread
                        .coverage
                        .entry("excluded_unresolved_own_counter".to_owned())
                        .or_default();
                    coverage.observations += 1;
                    coverage.counter_tokens += event.total_tokens;
                    continue;
                }
                event.session_id = key.clone();
                events.push(event);
            }
        }
    }
    let before_dedup = events.len();
    dedupe_codex_events(&mut events);
    let duplicate_events = before_dedup - events.len();
    filter_events_by_date(&mut events, shared)?;
    let mut units = BTreeMap::<String, Unit>::new();
    let mut priced = Vec::new();
    for event in &events {
        let unit = units.entry(event.session_id.clone()).or_default();
        let usage = Usage::event(event);
        unit.usage.add(&usage);
        unit.requests += 1;
        let model = if event.is_fallback_model {
            String::new()
        } else {
            crate::model_aliases::resolve_model_name(event.model.as_deref().unwrap_or(""))
                .into_owned()
        };
        let model_unit = unit.models.entry(model.clone()).or_default();
        model_unit.usage.add(&usage);
        let timestamp = crate::parse_ts_timestamp(&event.timestamp);
        let rate = timestamp.and_then(|timestamp| pricing.find_at(&model, timestamp));
        let category_reason = if event.cache_creation_tokens > 0 {
            Some("unpriced_cache_creation_rate")
        } else if event.cached_input_tokens > 0
            && rate.is_some_and(|rate| !rate.cache_read_explicit)
        {
            Some("unpriced_missing_explicit_cache_read_rate")
        } else {
            None
        };
        if let Some(reason) = category_reason {
            let coverage = threads
                .get_mut(&event.session_id)
                .expect("usage metadata exists")
                .coverage
                .entry(reason.to_owned())
                .or_default();
            coverage.observations += 1;
            coverage.counter_tokens += event.total_tokens;
        }
        if !model.is_empty() && rate.is_some() && category_reason.is_none() {
            unit.priced_tokens += event.total_tokens;
            model_unit.priced_tokens += event.total_tokens;
            let mut priced_event = event.clone();
            priced_event.model = Some(model);
            priced.push(priced_event);
        } else {
            unit.unpriced_tokens += event.total_tokens;
            model_unit.unpriced_tokens += event.total_tokens;
        }
    }
    let priced_groups = aggregate_events(
        &priced,
        AgentReportKind::Session,
        shared.timezone.as_deref(),
    )?;
    for (id, group) in priced_groups {
        let unit = units.get_mut(&id).expect("priced thread has usage");
        for (name, usage) in group.models {
            let cost =
                calculate_codex_model_cost(&name, &usage, pricing, resolve_codex_speed(speed));
            unit.known_cost += cost;
            unit.models
                .get_mut(&name)
                .expect("canonical priced model exists")
                .known_cost += cost;
        }
    }
    let mut thread_rows = Vec::new();
    let mut families = BTreeMap::<String, (Vec<String>, Unit)>::new();
    let mut workstream_units = BTreeMap::<String, (Vec<String>, Unit)>::new();
    let mut totals = Unit::default();
    let mut coverage_totals = BTreeMap::<String, Coverage>::new();
    for (id, thread) in &threads {
        let unit = units.get(id).cloned().unwrap_or_default();
        totals.add(&unit);
        let (root, cycle) = root_thread(id, &threads);
        let family = families.entry(root.clone()).or_default();
        family.0.push(id.clone());
        family.1.add(&unit);
        if let Some(label) = workstreams.get(id) {
            let target = workstream_units.entry(label.clone()).or_default();
            target.0.push(id.clone());
            target.1.add(&unit);
        }
        for (kind, value) in &thread.coverage {
            let target = coverage_totals.entry(kind.clone()).or_default();
            target.observations += value.observations;
            target.counter_tokens += value.counter_tokens;
            target.timestamps_missing += value.timestamps_missing;
        }
        thread_rows.push(json!({"threadId":id,"canonicalId":thread.canonical.then_some(id),"parentThreadId":thread.parent,"forkedFromId":thread.fork,"historyBaseThreadId":thread.history_base,"rootConversationId":root,"hierarchyUnresolved":cycle,"parentPresent":thread.parent.as_ref().map(|parent| threads.contains_key(parent)),"sourceSessions":thread.source_sessions,"workstream":workstreams.get(id),"ownUsage":unit.usage,"accounting":unit.json(),"coverage":thread.coverage}));
    }
    let family_rows=families.iter().map(|(root,(ids,unit))| {
        let root_unit=units.get(root).cloned().unwrap_or_default();
        let mut descendants=Unit::default();for id in ids {if id!=root {descendants.add(&units.get(id).cloned().unwrap_or_default());}}
        json!({"rootConversationId":root,"rootPresent":threads.contains_key(root),"threadIds":ids,"ownThreadUsage":root_unit.usage,"descendantUsage":descendants.usage,"totalUsage":unit.usage,"accounting":unit.json()})
    }).collect::<Vec<_>>();
    Ok(
        json!({"accountingVersion":1,"threads":thread_rows,"families":family_rows,"workstreams":workstream_units.iter().map(|(name,(ids,unit))| json!({"workstream":name,"threadIds":ids,"accounting":unit.json()})).collect::<Vec<_>>(),"totals":totals.json(),"coverage":{"duplicateEventsRemoved":duplicate_events,"observations":coverage_totals,"unmatchedWorkstreamThreadIds":workstreams.keys().filter(|id| !threads.contains_key(*id)).collect::<Vec<_>>()},"semantics":{"ownUsage":"Production parser increments after inherited replay exclusion and native event deduplication.","families":"Only explicit parent_thread_id and thread_spawn parent relationships group descendants. Fork links are reported separately.","coverage":"Counter tokens are observed baselines or anomalies, not additional usage. Observation counts can repeat across copied files.","cost":"Known model rates estimate API-equivalent cost. Unpriced tokens remain visible; costUSD is null for partial pricing.","workstreams":"Only explicitly mapped thread IDs contribute; mappings do not infer descendant membership.","dateFiltering":"Usage obeys the selected date range. Metadata and coverage describe all scanned native files."}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ccusage_test_support::fs_fixture;

    #[test]
    fn first_metadata_owns_thread_and_duplicate_snapshots_count_once() {
        let fixture = fs_fixture! {{
            "child.jsonl": concat!(
                "{\"type\":\"session_meta\",\"payload\":{\"id\":\"child\",\"parent_thread_id\":\"parent\"}}\n",
                "{\"type\":\"session_meta\",\"payload\":{\"id\":\"parent\"}}\n",
                "{\"type\":\"turn_context\",\"payload\":{\"model\":\"gpt-5\"}}\n",
                "{\"type\":\"event_msg\",\"timestamp\":\"2026-01-01T00:00:00Z\",\"payload\":{\"type\":\"token_count\",\"info\":{\"total_token_usage\":{\"input_tokens\":10,\"output_tokens\":1},\"last_token_usage\":{\"input_tokens\":10,\"output_tokens\":1}}}}\n",
                "{\"type\":\"event_msg\",\"timestamp\":\"2026-01-01T00:00:00Z\",\"payload\":{\"type\":\"token_count\",\"info\":{\"total_token_usage\":{\"input_tokens\":10,\"output_tokens\":1},\"last_token_usage\":{\"input_tokens\":10,\"output_tokens\":1}}}}\n"
            )
        }};
        let output = report_from_directory(
            fixture.root(),
            &SharedArgs::default(),
            CodexSpeed::Standard,
            &PricingMap::load_embedded(),
            None,
        )
        .unwrap();
        assert_eq!(output["threads"][0]["threadId"], "child");
        assert_eq!(output["threads"][0]["ownUsage"]["totalTokens"], 11);
        assert_eq!(output["threads"][0]["parentThreadId"], "parent");
        assert_eq!(output["families"][0]["rootPresent"], false);
        assert_eq!(
            output["coverage"]["observations"]["repeated_counter"]["observations"],
            1
        );
    }

    fn token(timestamp: &str, input: u64, cumulative_input: u64, cumulative_output: u64) -> Value {
        json!({"type":"event_msg","timestamp":timestamp,"payload":{"type":"token_count","info":{"last_token_usage":{"input_tokens":input,"output_tokens":1},"total_token_usage":{"input_tokens":cumulative_input,"output_tokens":cumulative_output}}}})
    }

    fn rollout(
        id: &str,
        parent: Option<&str>,
        fork: Option<&str>,
        model: Option<&str>,
        events: &[Value],
    ) -> String {
        let mut records = vec![
            json!({"type":"session_meta","payload":{"id":id,"parent_thread_id":parent,"forked_from_id":fork}}),
        ];
        if let Some(model) = model {
            records.push(json!({"type":"turn_context","payload":{"model":model}}));
        }
        records.extend_from_slice(events);
        records.iter().map(|record| format!("{record}\n")).collect()
    }

    #[test]
    fn explicit_nested_parents_group_families_without_grouping_forks_or_copies() {
        let parent = rollout(
            "parent",
            None,
            None,
            Some("gpt-5"),
            &[token("2026-01-01T00:00:00Z", 10, 10, 1)],
        );
        let child = rollout(
            "child",
            Some("parent"),
            None,
            Some("gpt-5"),
            &[token("2026-01-01T00:01:00Z", 20, 20, 1)],
        );
        let nested = rollout(
            "nested",
            Some("child"),
            None,
            Some("gpt-5"),
            &[token("2026-01-01T00:02:00Z", 30, 30, 1)],
        );
        let fork = rollout(
            "fork",
            None,
            Some("parent"),
            Some("gpt-5"),
            &[
                token("2026-01-01T00:00:00Z", 10, 10, 1),
                token("2026-01-01T00:03:00Z", 40, 50, 2),
            ],
        );
        let fixture = fs_fixture!({"parent.jsonl":parent.clone(),"copy.jsonl":parent,"child.jsonl":child,"nested.jsonl":nested,"fork.jsonl":fork});
        let output = report_from_directory(
            fixture.root(),
            &SharedArgs::default(),
            CodexSpeed::Standard,
            &PricingMap::load_embedded(),
            None,
        )
        .unwrap();
        let families = output["families"].as_array().unwrap();
        let family = families
            .iter()
            .find(|row| row["rootConversationId"] == "parent")
            .unwrap();
        assert_eq!(family["ownThreadUsage"]["totalTokens"], 11);
        assert_eq!(family["descendantUsage"]["totalTokens"], 52);
        assert_eq!(family["totalUsage"]["totalTokens"], 63);
        assert_eq!(output["totals"]["usage"]["totalTokens"], 104);
        assert_eq!(output["coverage"]["duplicateEventsRemoved"], 1);
        assert_eq!(
            output["coverage"]["observations"]["inherited_replay"]["observations"],
            1
        );
    }

    #[test]
    fn unknown_models_and_rates_remain_unpriced_and_gaps_remain_coverage() {
        let unknown = rollout(
            "unknown",
            None,
            None,
            None,
            &[token("2026-01-01T00:00:00Z", 10, 10, 1)],
        );
        let unsupported = rollout(
            "unsupported",
            None,
            None,
            Some("unknown-model-for-test"),
            &[token("2026-01-01T00:01:00Z", 20, 20, 1)],
        );
        let gap = rollout(
            "gap",
            None,
            None,
            Some("gpt-5"),
            &[
                token("2026-01-01T00:02:00Z", 10, 10, 1),
                token("2026-01-01T00:03:00Z", 5, 30, 2),
                token("2026-01-01T00:04:00Z", 3, 3, 1),
            ],
        );
        let fixture =
            fs_fixture!({"unknown.jsonl":unknown,"unsupported.jsonl":unsupported,"gap.jsonl":gap});
        let output = report_from_directory(
            fixture.root(),
            &SharedArgs::default(),
            CodexSpeed::Standard,
            &PricingMap::load_embedded(),
            None,
        )
        .unwrap();
        assert_eq!(output["totals"]["unpricedTokens"], 32);
        assert!(output["totals"]["costUSD"].is_null());
        assert_eq!(output["totals"]["usage"]["totalTokens"], 53);
        assert_eq!(
            output["coverage"]["observations"]["cumulative_gap"]["observations"],
            1
        );
        assert_eq!(
            output["coverage"]["observations"]["counter_reset"]["observations"],
            1
        );
    }

    #[test]
    fn only_explicit_workstream_members_are_aggregated() {
        let parent = rollout(
            "parent",
            None,
            None,
            Some("gpt-5"),
            &[token("2026-01-01T00:00:00Z", 10, 10, 1)],
        );
        let child = rollout(
            "child",
            Some("parent"),
            None,
            Some("gpt-5"),
            &[token("2026-01-01T00:01:00Z", 20, 20, 1)],
        );
        let fixture = fs_fixture!({"logs/parent.jsonl":parent,"logs/child.jsonl":child,"mapping.json":r#"{"parent":"PR-123"}"#});
        let mapping = fixture.path("mapping.json");
        let output = report_from_directory(
            &fixture.path("logs"),
            &SharedArgs::default(),
            CodexSpeed::Standard,
            &PricingMap::load_embedded(),
            mapping.to_str(),
        )
        .unwrap();
        assert_eq!(
            output["workstreams"][0]["accounting"]["usage"]["totalTokens"],
            11
        );
        assert_eq!(output["families"][0]["totalUsage"]["totalTokens"], 32);
    }

    #[test]
    fn unresolved_initial_cumulative_without_last_is_excluded() {
        let records = rollout(
            "unknown-baseline",
            None,
            None,
            Some("gpt-5"),
            &[
                json!({"type":"event_msg","timestamp":"2026-01-01T00:00:00Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":40,"output_tokens":4}}}}),
            ],
        );
        let fixture = fs_fixture!({"session.jsonl":records});
        let output = report_from_directory(
            fixture.root(),
            &SharedArgs::default(),
            CodexSpeed::Standard,
            &PricingMap::load_embedded(),
            None,
        )
        .unwrap();
        assert_eq!(output["totals"]["usage"]["totalTokens"], 0);
        assert_eq!(
            output["coverage"]["observations"]["excluded_unresolved_own_counter"]["counterTokens"],
            44
        );
    }

    #[test]
    fn cache_creation_requests_have_visible_unpriced_coverage() {
        let mut event = token("2026-01-01T00:00:00Z", 10, 10, 1);
        event["payload"]["info"]["last_token_usage"]["cache_write_input_tokens"] = json!(2);
        event["payload"]["info"]["total_token_usage"]["cache_write_input_tokens"] = json!(2);
        let records = rollout("cache-write", None, None, Some("gpt-5"), &[event]);
        let fixture = fs_fixture!({"session.jsonl":records});
        let output = report_from_directory(
            fixture.root(),
            &SharedArgs::default(),
            CodexSpeed::Standard,
            &PricingMap::load_embedded(),
            None,
        )
        .unwrap();
        assert_eq!(output["totals"]["unpricedTokens"], 11);
        assert_eq!(
            output["coverage"]["observations"]["unpriced_cache_creation_rate"]["counterTokens"],
            11
        );
        assert!(output["totals"]["costUSD"].is_null());
    }
}
