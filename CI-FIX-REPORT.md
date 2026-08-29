# CI Fix Report

Date: 2026-08-29
Repository: `hermes-gadget/Raven-Agent`
Branch: `master`

## Failed run investigated

- Workflow: `CI`
- Run ID: `32659164473`
- Commit: `cf3896cbb214b14d83820f87db1cb0d9f409bdad`

## Root causes

### Workspace compilation

`DiscordConfig` gained an `agent_selector` field, but two explicit test
initializers in `crates/odin-gateway/src/discord.rs` were not updated. Rust
reported `E0063` for the missing field in `test_start_without_token` and
`test_start_with_mock_token_signals_connected`.

This one compilation error caused all target-building jobs to fail:

- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `cargo bench --no-run`
- `scripts/validate-tools.sh` at its full-workspace test step

### Security audit

The audit job had the required `contents: read` and `checks: write`
permissions. Its current failure was dependency data in `Cargo.lock`, not a
GitHub token permission error.

- `h2 0.4.15` was affected by `RUSTSEC-2026-0258` (unbounded empty DATA
  frames; patched in `>= 0.4.16`).
- `event-listener 5.4.1` was affected by the unsoundness advisory
  `RUSTSEC-2026-0221` (patched in `>= 5.4.2`).
- `lru 0.18.1` was affected by the unsoundness advisory
  `RUSTSEC-2026-0253` (patched in `>= 0.18.2`).
- `chacha20 0.10.1` and `spin 0.9.8` were yanked.

## Fix applied

- Added `agent_selector: None` to the two stale `DiscordConfig` test
  initializers.
- Updated only compatible transitive lockfile releases; no direct dependency
  was added:
  - `h2 0.4.15 -> 0.4.19`
  - `event-listener 5.4.1 -> 5.4.2`
  - `lru 0.18.1 -> 0.18.3`
  - `chacha20 0.10.1 -> 0.10.2`
  - `spin 0.9.8 -> 0.9.9`

## Local verification

All commands passed on Rust 1.98.0:

- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets`
- `bash scripts/validate-tools.sh`
- `cargo bench --no-run`
- `cargo fmt --all -- --check`
- `cargo audit` (no vulnerabilities or warnings)
- `git diff --check`

Remote CI is triggered by the commit containing this report and is verified
against the resulting `master` workflow run after the push.
