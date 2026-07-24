# Raven Agent architecture

This document describes the implemented 0.3.0 architecture. Internal crate names retain the historical **odin-\*** prefix; external naming uses Raven Agent and the **raven** command.

## Runtime paths

~~~text
                             +------------------+
                             | model providers  |
                             +---------+--------+
                                       |
raven run ---> composer ---> task graph + sub-agents ---> merge ---> result
    |              |              |          |
    |              |              |          +-- lifecycle + audit
    |              |              +------------- file locks + snapshots
    |              +---------------------------- SQLite graph state
    |
    +-------- direct runtime agent ---> seven-phase loop

raven / TUI ------ in-process runner + SQLite monitoring + run controls
raven serve ------ HTTP /chat + /orchestrate + /ws
Discord ---------- /raven run + persisted orchestration inspection
scheduler -------- persisted definitions + runtime dispatch when hosted
~~~

The execution loop is:

~~~text
PLAN -> ACT -> INSPECT -> CRITIQUE -> REVISE -> VERIFY -> DECIDE
  ^                                                        |
  +----------------------- continue ------------------------+
~~~

Without a provider, the loop uses deterministic heuristic behavior for tests and offline diagnostics. Production execution supplies a configured provider.

## Crates

| Internal crate | Responsibility |
|---|---|
| odin-core | Shared types, YAML configuration, errors, and traits |
| odin-cli | Shared command implementation plus raven and legacy odin entry points |
| odin-loop | Seven-phase model loop, tool dispatch, policy enforcement, skills, and audit hooks |
| odin-eval | Deterministic small-model eval suite, model-profile reports, and live-eval readiness gates |
| odin-orchestrator | Goal decomposition, graphs, sub-agent state, locks, merge, and SQLite persistence |
| odin-runtime | Agents, sessions, and memory-backed task submission |
| odin-providers | OpenAI-compatible, Anthropic, local, fallback, health, and circuit-breaker providers |
| odin-tools | Built-in registry, sandbox boundaries, validation, catalog, dry-run, and reliability model |
| odin-mcp | MCP stdio client and external-tool adapter |
| odin-permissions | Policy rules, rate limits, approval decisions, secrets, and secret/PII redaction |
| odin-audit | Redacted buffered and JSONL audit logging |
| odin-memory | SQLite memory store |
| odin-scheduler | Cron parsing, job state, SQLite persistence, and optional runtime dispatch |
| odin-gateway | HTTP, WebSocket, and Discord adapters |
| odin-tui | Chat-first terminal UI, in-process runner, live state rendering, and run controls |
| odin-skills | Markdown skill loading and tool dependency checks |
| odin-baseline | Naive comparison implementation used by benchmarks and tests |

## Orchestration identity and persistence

Every task graph owns a UUID. That UUID is the run ID returned by CLI, HTTP, Discord, and TUI interfaces and the primary key in the orchestration database. Legacy databases can still resolve graphs by their old goal-text key.

The store persists:

- graph structure, node state, results, and overall status;
- agent lifecycle transitions;
- the latest file-lock and write-queue snapshot.

**raven run** and TUI-submitted runs write these records during real execution. **orchestrate submit**, HTTP **/orchestrate**, and Discord **/raven orchestrate submit** create a building-state plan without claiming execution.

Stored pause/resume/cancel changes from the non-interactive **orchestrate** commands are state markers. They are not a cross-process control channel. The TUI can pause, resume, redirect, reprioritise, and cancel the in-process run that it started.

## Tool and permission path

~~~text
model tool call
    |
    +-- registry lookup and JSON argument parsing
    +-- per-agent scoped registry
    +-- rate limit
    +-- explicit allow / deny / ask rule
    +-- tool approval metadata
    +-- dangerous shell-command check
    +-- filesystem path boundary
    +-- execute
    +-- redact result and error
    +-- append redacted audit entry
~~~

Approval-required calls fail closed when no responder is connected. Direct tool testing requires **--approve** for dangerous tools. MCP tools default to unsafe and approval-required.

The sandbox currently enforces path boundaries in tool implementations. The **sandbox_enabled** configuration field does not create a container or chroot.

## Interfaces

### CLI and TUI

The primary Cargo target and installed command are **raven**. Running it without a subcommand opens the TUI. The **odin** binary calls the same command implementation but requires a subcommand.

The TUI reads the orchestration database every tick and overlays in-memory runner events for sub-second feedback. It shows real graph, agent, lock, catalog, log, history, and conflict data. Submitting chat starts an in-process orchestration runner that uses the configured provider fallback chain, permission policy, built-in and MCP tool registry, redacted audit logger, and provider HTTP timeouts. The runner emits stages for decomposition, resource loading, agent spawn, lock waits, model waits, tool requests, failures, and completion. The UI displays immediate chat feedback, agent heartbeats, elapsed model waits after 10 seconds, stale-event warnings after 15 seconds, and visible errors/blockers. Follow-up chat messages steer the active run; `/pause`, `/resume`, `/redirect`, `/prio`, and approval-gated `/cancel` target that same run.

### HTTP and WebSocket

The HTTP server exposes health, chat/task, live-registry tool inspection, orchestration plan/state, and WebSocket routes. The **/chat** handler builds a runtime agent with the configured provider, tools, permission engine, memory store, and audit logger.

WebSocket output events are broadcast through the shared connection manager. Inbound pause/resume/cancel messages are parsed but not dispatched to an executor.

### Discord

When enabled in gateway configuration, **raven serve** starts and gracefully stops Discord alongside HTTP. The default slash-command root is **/raven**; an explicit legacy **odin** prefix remains configurable. Run commands submit to the configured runtime. Orchestration commands create or inspect persisted plans and state.

### Scheduler

CLI scheduler commands load and persist definitions in SQLite on every invocation. **raven schedule host** keeps the scheduler, SQLite store, runtime agent, memory, tools, and audit logger alive until SIGINT/SIGTERM. It starts only when `scheduler.enabled` is true.

The host coalesces missed cron occurrences into one immediate execution after restart. Per-job overlap defaults to one execution; an overlapping due occurrence remains pending until the prior run finishes. `scheduler.max_concurrent` bounds total host executions. Shutdown stops new dispatch immediately and waits for in-flight jobs. Starts and terminal outcomes are written to the audit log, while durable outcomes in `scheduler_runs` back **raven schedule status** and **raven schedule history**.

## Data paths

Canonical configuration: **~/.config/raven/config.yaml**

Default data directory: **~/.raven-agent**

- memory.db
- scheduler.db
- orchestration.db
- audit.jsonl

Configured memory, audit, scheduler, and general data paths override those defaults. Old config paths and ODIN_CONFIG remain read fallbacks.

## Boundaries still open

The authoritative list is in [TODO.md](TODO.md). The main architectural gaps are general interactive tool-call approvals, cross-process orchestration control, inbound WebSocket control dispatch, orchestrated memory retrieval, continuous scheduler hosting, and persistent reliability samples.
