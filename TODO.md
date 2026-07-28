# Raven Agent — TODO

> Updated: 2026-07-09 | Workspace version: 0.3.0

## Full audit pass (2026-07-09)

See [`audit.md`](audit.md) for the complete checklist, root causes, and verification.

- [x] CLI multi-task `get_graph` goal-shadowing panic fixed
- [x] CLI no longer executes agents that failed file-lock acquisition
- [x] Sub-agent tool scoping from capabilities (not full registry by default)
- [x] TUI runner JoinError cannot hang a run forever
- [x] TUI Error event clears active-run handles
- [x] Pause messaging documents in-flight work may finish
- [x] TUI loads provider/model into top bar from config
- [x] All required cargo/script gates green

## TUI Live Execution Bugfix

Treat this as complete only after reproducing the current TUI behavior from `raven`, tracing the real submit → orchestration → event → render path, and verifying visible live progress under failure and slow-provider cases.

### Required debug

- [x] Run `raven`.
- [x] Start a real task from the TUI.
- [x] Trace input box → submit handler → orchestration run created → sub-agents spawned → provider/tool calls → events emitted → TUI receives events → render update.
- [x] Identify where live progress can stop or confirm the current path is wired end-to-end.
- [x] Add or verify logging/event tracing at every stage.

### Required fixes

- [x] No silent task starts; within 1 second the UI shows a run ID and status.
- [x] Chat panel immediately shows submitted, planning, decomposing, spawning agents, running, waiting for model/tool/lock/approval, failed, and done states.
- [x] Side panel shows live sub-agents, current task, lifecycle state, elapsed time, current tool/model call, locks, queued writes, and last event.
- [x] Provider calls over 10 seconds show `waiting for model...` with elapsed timer.
- [x] No event for 15 seconds shows a `no events received` warning with last known stage.
- [x] Orchestration startup failures show the real UI error.
- [x] TUI is connected to the real orchestrator/event bus, not mock/static state except in tests.
- [x] User interrupt/cancel targets the active run.

### Tests

- [x] Fake slow provider test proves UI keeps updating.
- [x] TUI submit → run created → planning event → agent event → completion event.
- [x] Provider unavailable shows visible error.
- [x] Cancel works during model wait.
- [x] Five unrelated tasks show five agents in the side panel.

### Definition of done

- [x] Starting a task in `raven` always shows visible progress within 1 second.
- [x] TUI streams real orchestration events and shows agents working.
- [x] TUI surfaces blockers/errors instead of leaving a dead screen.

## Small-Model Excellence Phase

Only mark these complete after Raven is measured against repeatable baselines, adapts execution to model capability, and docs report which small/local/cheap models work best and why.

### Phase 6 blockers that affect this phase

- [x] Re-audit real persistent orchestration, restart recovery, real status/pause/resume/cancel, scoped sub-agents, MCP startup wiring, and full verification against the current worktree.
- [x] Fix any blocker from that audit before relying on the small-model eval results. No blocking gaps found in the current audit.

### Eval harness and baselines

- [x] Add a repeatable small-model eval harness.
- [x] Cover coding, repo edits, debugging, docs, tool use, multi-file work, long context, and failed tool recovery.
- [x] Compare Raven looped/orchestrated execution against a single-pass baseline on the same tasks.
- [x] Run mocked evals in CI.
- [x] Make live provider evals optional and gated by provider keys/config.
- [x] Produce a scoring dashboard/report with success rate, cost, tokens, iterations, tool errors, time, and escalation rate.

### Small/local model support

- [x] Test Ollama/OpenAI-compatible local model profiles through mocked evals; live provider eval remains opt-in through `raven eval live`.
- [x] Test small/cheap DeepSeek-compatible profiles through mocked evals; live provider eval remains opt-in through `raven eval live`.
- [x] Test Qwen and Llama model families through mocked evals where available.
- [x] Add model profiles for context size, strengths, weaknesses, max tool complexity, preferred prompt style, retry limits, and escalation rules.
- [x] Document which small models work best and why.

### Adaptive execution

- [x] Keep simple tasks cheap.
- [x] Escalate hard tasks to decomposition, sub-agents, stronger verifier, or stronger model only when needed.
- [x] Add retry/escalation rules driven by model profile and failure evidence.

### Small-model prompting and structure

- [x] Improve prompts for small models with short instructions, strict schemas, examples, bounded outputs, and tool-choice hints.
- [x] Add structured JSON planning/decomposition with validation and fallback.
- [x] Add tool-use repair that auto-corrects invalid arguments once before retry.
- [x] Add context distillation into task facts, decisions, files changed, errors, and next action.
- [x] Add verifier/judge mode that checks real evidence instead of self-confidence only.
- [x] Add failure taxonomy for model confusion, bad tool args, missing context, permission denied, timeout, and hallucinated file/tool.

### Required gates

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo check --workspace --all-targets`
- [x] `cargo test --workspace --all-targets`
- [x] `scripts/validate-tools.sh`

## TUI Responsiveness + Live Agent Feedback Debug

Only mark these items complete after reproducing the stall, tracing the live path, implementing the fix, and verifying current runtime behavior.

### Required investigation

- [x] Reproduce the TUI responsiveness issue by running `raven` and starting a real orchestrated task.
- [x] Confirm whether the orchestrator is running, blocked, waiting for provider, waiting for approval, waiting for locks, or failed.
- [x] Trace TUI input -> run creation -> composer -> sub-agent spawn -> provider/tool calls -> event stream -> TUI render.
- [x] Add logs/telemetry around each runtime stage.
- [x] Identify why the UI can show no useful feedback for long periods.

### Required fixes

- [x] Show immediate feedback within 1 second after task submission.
- [x] Surface stages: planning, decomposing, spawning agents, waiting for model, running tool, waiting for lock, approval needed, retrying, failed, done.
- [x] Add visible spinner/heartbeat for active agents.
- [x] Update the right-side agent panel live with status, current task, current tool/model call, elapsed time, and last event.
- [x] Show progress messages in the chat panel.
- [x] Show `waiting for model...` with elapsed time when a provider/model call exceeds 10 seconds.
- [x] Warn when no event arrives for 15 seconds and show the last known stage.
- [x] Show exact blocked reasons and runtime errors in the UI.
- [x] Add timeout handling and UI cancellation that actually stops the active run.
- [x] Ensure user interrupts steer or cancel the active run, not disconnected state.

### Testing

- [x] Add automated TUI state/event update tests where practical.
- [x] Add a fake slow provider test using mocked time/events to prove the UI keeps updating during a 90s-equivalent wait.
- [x] Add submit -> planning -> agent running -> tool/model wait -> result integration coverage.
- [x] Manually test over SSH/small terminal.
- [x] Test one task and five unrelated tasks.
- [x] Test provider unavailable, slow provider, tool failure, approval needed, lock wait, cancel, pause/resume.

### Required gates

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo check --workspace --all-targets`
- [x] `cargo test --workspace --all-targets`
- [x] `scripts/validate-tools.sh`

## End-to-End Verification + TUI Rework

Only mark these items complete when they are implemented, wired to real state, tested, documented, and usable.

### TUI requirements

- [x] Make `raven` launch the primary TUI by default.
- [x] Rework the TUI into a chat-first, keyboard-driven Codex/Claude Code-style interface.
- [x] Keep the main center/left panel focused on conversation, goals, Raven responses, approval prompts, errors, and final summaries.
- [x] Keep a multiline input box fixed at the bottom.
- [x] Keep a live agent status panel always visible by default on the right.
- [x] Show active run ID, sub-agent names, agent tasks, lifecycle state, current tool call, files read/written, held locks, queued write locks, progress %, and last update/error in the right panel.
- [x] Add tabs for Chat, Agents, Task Graph, Files/Locks, Tools, Logs/Audit, History, and Conflicts.
- [x] Keep logs/audit detail out of the chat view except for relevant summaries/errors.
- [x] Show dangerous actions as clear approve/deny modals.
- [x] Route user messages into the active run instead of creating disconnected fake runs.
- [x] Support pause, resume, cancel, redirect, and reprioritise interruptions.
- [x] Stream live updates from real orchestration state.
- [x] Ensure the UI works over SSH and small terminals.

### End-to-end verification

- [x] Test `raven`.
- [x] Test `raven run`.
- [x] Test `raven run --direct`.
- [x] Test `raven ui`.
- [x] Test orchestration with 5 unrelated tasks.
- [x] Test overlapping file edits queue correctly.
- [x] Test pause/resume/cancel.
- [x] Test restart recovery.
- [x] Test provider fallback.
- [x] Test MCP/tool loading.
- [x] Test approval gates and secret/PII redaction.
- [x] Test TUI rendering/event handling/state updates.

### Required gates

- [x] `cargo fmt --all -- --check`
- [x] `cargo clippy --workspace --all-targets -- -D warnings`
- [x] `cargo check --workspace --all-targets`
- [x] `cargo test --workspace --all-targets`
- [x] `cargo bench --no-run` where relevant
- [x] `scripts/validate-tools.sh`

## Repo Tidy + Stabilisation Pass — Complete

### Naming, version, and packaging

- [x] Make **Raven Agent** the user-facing product name.
- [x] Make **raven** the primary and default command.
- [x] Keep **odin** only as an internal crate prefix and documented compatibility alias.
- [x] Separate the raven and odin binary entry points so Cargo no longer builds one source file as two targets.
- [x] Use version **0.3.0** across manifests, runtime output, README, CHANGELOG, docs, examples, and MCP client metadata.
- [x] Point repository metadata and quickstart at the real GitHub repository. The historical repository slug is intentionally retained as a locator.

### CLI, TUI, and configuration

- [x] Make bare **raven** open the TUI; keep bare **odin** non-interactive.
- [x] Replace disconnected TUI approval, pause, resume, interrupt, and cancel controls.
- [x] Make TUI submission persist a real building-state plan with a stable run ID.
- [x] Populate TUI graph, agents, locks, tool catalog, redacted log, history, progress counts, and declared conflicts from real data.
- [x] Fix Unicode input cursor and deletion boundaries.
- [x] Use **~/.config/raven/config.yaml** and **RAVEN_CONFIG** canonically while retaining legacy read fallbacks.
- [x] Wire configured data, memory, audit, scheduler, sandbox, enabled-tool, disabled-tool, and MCP environment settings.
- [x] Make missing explicit config paths fail with an actionable error instead of silently using defaults.
- [x] Replace the stale annotated config with the actual deserializable schema.

### Orchestration, scheduler, tools, and gateways

- [x] Persist graphs under the same UUID returned by CLI, HTTP, Discord, and TUI.
- [x] Retain legacy goal-key graph lookup for old databases.
- [x] Keep graph JSON and summary statuses synchronized, including paused state.
- [x] Persist real CLI run node transitions, agent lifecycles, and lock snapshots.
- [x] Make Composer update graph node assignment, status, result, and final graph status as agents change.
- [x] Enforce configured tool allow/disable lists and real sub-agent allow-list scoping.
- [x] Use one standard built-in registry for execution, TUI, and diagnostics.
- [x] Wire tool registry and audit logging into direct, orchestrated, and HTTP loop engines.
- [x] Make HTTP health, metrics, inspection, validation, and doctor endpoints use the live configured registry.
- [x] Mount the advertised **/ws** endpoint with the shared broadcast manager.
- [x] Start and stop Discord from **raven serve** when gateway configuration enables it.
- [x] Make scheduler CLI definitions and task goals survive separate invocations.
- [x] Dispatch scheduled runtime jobs to a registered agent and fail clearly instead of running an inert closure when runtime wiring is missing.
- [x] Add continuously hosted scheduler execution with restart recovery, bounded overlap, graceful shutdown, and durable outcomes.
- [x] Pass configured MCP environment variables to child processes.
- [x] Make unknown MCP tools unsafe and approval-required by default.
- [x] Remove generated reliability scores and other demonstration status output.
- [x] Replace fake Discord direct-send success with an actionable unsupported-operation error.
- [x] Make Discord default to **/raven** while allowing an explicitly configured legacy prefix.

### Safety, redaction, and tests

- [x] Enforce rate limits, permission decisions, approval requirements, command checks, and path boundaries before tool execution.
- [x] Fail closed when approval is required and no responder is connected.
- [x] Require **--approve** for direct dangerous-tool execution; retain **--dry-run** validation.
- [x] Replace successful simulated tool dispatch with a real failure when no registry is configured.
- [x] Redact supported secrets and PII from tool results, direct tool output, TUI logs, config display, audit reads, and audit writes.
- [x] Make durable audit redaction unconditional, including for legacy **mask_secrets: false** config.
- [x] Remove raw goals, commands, message bodies, and paths from diagnostic tracing where found.
- [x] Replace the always-true baseline assertion with a behavioral assertion.
- [x] Fix the circuit-breaker test deadlock and add coverage for stable run IDs, synchronized status, scoped registries, real TUI plan persistence, and Unicode input.
- [x] Remove the tracked temporary source file and unused fields/helpers found by strict all-target linting.

### Documentation

- [x] Replace README quickstart, command list, architecture, safety behavior, and limitations with current behavior.
- [x] Replace stale architecture, tools, compatibility, and comparison claims.
- [x] Clearly distinguish real-model behavior from deterministic comparison fixtures.
- [x] Update CHANGELOG for this pass.

### Required gates

- [x] **cargo fmt --all -- --check**
- [x] **cargo clippy --workspace --all-targets -- -D warnings**
- [x] **cargo check --workspace --all-targets**
- [x] **cargo test --workspace --all-targets**
- [x] **cargo bench --no-run**
- [x] **scripts/validate-tools.sh**

## Deferred work

These items are intentionally not represented as complete:

- [ ] [#1 Interactive approval responder](https://github.com/hermes-gadget/raven-ai-harness/issues/1)
- [ ] [#4 Live cross-process orchestration control and WebSocket dispatch](https://github.com/hermes-gadget/raven-ai-harness/issues/4)
- [ ] [#5 Orchestrated sub-agent memory integration](https://github.com/hermes-gadget/raven-ai-harness/issues/5)
- [x] [#3 Continuously hosted scheduler execution mode](https://github.com/hermes-gadget/Raven-Agent/issues/3)
- [ ] [#2 Persisted real tool reliability samples](https://github.com/hermes-gadget/raven-ai-harness/issues/2)

## Definition of done

- [x] Product naming, command naming, version, config paths, and repository metadata are consistent.
- [x] No known placeholder UX or fabricated status output remains in user-facing paths.
- [x] Execution paths use real registries, persistence, permissions, redaction, and audit hooks.
- [x] README and architecture state limitations instead of presenting deferred behavior as complete.
- [x] Genuine deferrals have focused GitHub issues.
- [x] Every required verification gate passes on the final worktree.
