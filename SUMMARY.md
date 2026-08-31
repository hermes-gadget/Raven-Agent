# Raven Audit Recommendations 1–2

## Outcome

Implemented the first two recommendations from `AUDIT_REPORT.md` in place on branch `impl/raven-audit`. Existing CLI, TUI, HTTP, Discord, scheduler, and orchestration flows remain intact and now share the same validated production composition.

## 1. Evidence-based completion

- A task cannot report success unless at least one ACT phase ran, the plan reached verified completion, observable evidence exists, and final confidence meets the configured threshold.
- Verification now applies the evidence gate for every engine, not only engines with a small-model profile.
- Placeholder ACT output, provider errors, tool-call announcements, and response length do not count as completion evidence.
- The loop distinguishes verified completion, iteration exhaustion, ordinary stop, and unavailable escalation.
- Iteration exhaustion always returns a useful summary containing the action/tool evidence gathered and any remaining work. A one-iteration task reports that planning completed before ACT instead of claiming success.
- Successful results return the substantive ACT result or concrete tool evidence instead of the old generic counters-only summary.

## 2. Validated production composition root

- Added `odin_runtime::ProductionComposition`, the single builder used by direct CLI, CLI orchestration, TUI orchestration, HTTP, Discord, and scheduler host execution.
- Startup validation rejects a missing default provider, a missing/empty model, invalid or duplicate fallback references, self-fallback, invalid confidence thresholds, zero configured iterations, and empty phase-model settings.
- The builder resolves the primary model from `models.default_model` or the selected provider's `default_model`, constructs the configured fallback chain once, and shares it across all agents/requests.
- Planning, critique/verification, and escalation model settings are now wired into their corresponding loop phases. Matching built-in small-model profiles and configured skills are also attached consistently.
- Shared policy, tool registry, audit logger, and reliability tracker resources are applied by the same builder. Per-agent scoped tools and TUI progress wrappers remain supported as explicit execution overrides.
- Composed agents use one stable ID for the runtime agent and engine security principal.

## Tests and verification

- Added regressions proving one PLAN-only iteration cannot succeed and iteration exhaustion returns completed work.
- Added composition tests covering all six production surfaces, provider-default model resolution, fallback validation, and consistent phase-model routing.
- Updated deterministic eval/comparison fixtures to provide explicit ACT and verification evidence rather than relying on offline placeholders.
- `cargo test --workspace` — passed (configured live/optional tests remained ignored).
- `cargo check --workspace --all-targets` — passed.
- `cargo fmt --all -- --check` — passed.
- `cargo clippy --workspace --all-targets -- -D warnings` — passed.

No services were restarted, and no deployment or push was performed.
