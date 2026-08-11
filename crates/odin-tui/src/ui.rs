//! Clean Raven TUI rendering.
//! Layout: top bar (1 line), main (chat 65% | side 35%), bottom bar (1 line).
//! Approval prompts and help appear as centered modals. No emoji, pure ASCII.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
};

use super::app::{self, App, LogLevel, MessageRole, Panel, RunMode};

// ── Colors ───────────────────────────────────────────────────────────

const BG: Color = Color::Rgb(22, 24, 38);
const SURFACE: Color = Color::Rgb(30, 33, 48);
const FG: Color = Color::Rgb(212, 216, 236);
const ACCENT: Color = Color::Rgb(137, 180, 250);
const GREEN: Color = Color::Rgb(166, 227, 161);
const YELLOW: Color = Color::Rgb(249, 226, 175);
const RED: Color = Color::Rgb(243, 139, 168);
const MUTED: Color = Color::Rgb(88, 91, 112);
const DIM: Color = Color::Rgb(49, 50, 68);
const CYAN: Color = Color::Rgb(148, 226, 213);
const MAGENTA: Color = Color::Rgb(203, 166, 247);

// ── Phase markers — pure ASCII ──────────────────────────────────────

pub fn phase_icon(phase: &str) -> (&str, Color) {
    match phase {
        "running" => ("[>]", GREEN),
        "planning" | "decomposing" | "spawning_agents" => ("[*]", ACCENT),
        "waiting_for_model" => ("[M]", YELLOW),
        "running_tool" => ("[T]", CYAN),
        "approval_needed" => ("[A]", YELLOW),
        "retrying" => ("[R]", YELLOW),
        "done" | "complete" => ("[v]", GREEN),
        "failed" => ("[X]", RED),
        "paused" => ("[||]", YELLOW),
        "cancelled" | "canceled" => ("[=]", RED),
        "queued" | "pending" => ("[.]", MUTED),
        "blocked" => ("[B]", YELLOW),
        "building" => ("[*]", ACCENT),
        _ => ("[?]", MUTED),
    }
}

pub fn mode_str(mode: RunMode) -> (&'static str, Color) {
    match mode {
        RunMode::Idle => ("IDLE", MUTED),
        RunMode::Running => ("RUNNING", GREEN),
        RunMode::Paused => ("PAUSED", YELLOW),
    }
}

// ── Top Bar ──────────────────────────────────────────────────────────

fn render_top_bar(frame: &mut Frame, area: Rect, app: &App) {
    let run_id = app
        .active_run_id
        .as_deref()
        .or_else(|| app.running_runs.first().map(String::as_str))
        .map(|r| r.chars().take(8).collect::<String>())
        .unwrap_or_else(|| "-".into());
    let (mode_text, mode_color) = mode_str(app.mode);
    let provider = if app.provider_name.is_empty() {
        "-"
    } else {
        &app.provider_name
    };
    let model = if app.model_name.is_empty() {
        "-"
    } else {
        &app.model_name
    };

    let line = Line::from(vec![
        Span::styled(
            " Raven Agent ",
            Style::default()
                .fg(BG)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" ", Style::default()),
        Span::styled(format!("run:{}", run_id), Style::default().fg(CYAN)),
        Span::styled("  ", Style::default()),
        Span::styled(
            format!("{}/{}", provider, model),
            Style::default().fg(MUTED),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(
            format!("[{}]", mode_text),
            Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ? help", Style::default().fg(DIM)),
    ]);

    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(SURFACE)),
        area,
    );
}

// ── Bottom Bar ───────────────────────────────────────────────────────

fn render_bottom_bar(frame: &mut Frame, area: Rect, app: &App) {
    let elapsed = app.elapsed.as_secs();
    let line = Line::from(vec![
        Span::styled(
            format!("tools:{} ", app.tool_call_count),
            Style::default().fg(CYAN),
        ),
        Span::styled(
            format!("locks:{} ", app.orch.locks.len()),
            Style::default().fg(YELLOW),
        ),
        Span::styled(
            format!("writes:{} ", app.orch.write_queue.len()),
            Style::default().fg(MAGENTA),
        ),
        Span::styled(
            format!("errors:{} ", app.error_count),
            Style::default().fg(if app.error_count > 0 { RED } else { MUTED }),
        ),
        Span::styled(format!("elapsed:{}s ", elapsed), Style::default().fg(MUTED)),
        Span::styled(
            if app.last_action.is_empty() {
                ""
            } else {
                &app.last_action
            },
            Style::default().fg(DIM),
        ),
    ]);

    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(SURFACE)),
        area,
    );
}

// ── Chat Panel ───────────────────────────────────────────────────────

fn render_chat_panel(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" Chat ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(MUTED))
        .style(Style::default().bg(BG));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(2), Constraint::Length(4)])
        .split(inner);

    // Messages
    let max_msgs = chunks[0].height as usize;
    let visible: Vec<&app::ChatMessage> = app.messages.iter().rev().take(max_msgs).collect();
    let mut lines: Vec<Line> = Vec::new();

    for msg in visible.iter().rev() {
        let (prefix, color) = match msg.role {
            MessageRole::User => (">", ACCENT),
            MessageRole::Agent => ("<", GREEN),
            MessageRole::System => ("-", MUTED),
        };

        for (i, cl) in msg.content.lines().enumerate() {
            if i == 0 {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{} ", prefix),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(cl, Style::default().fg(FG)),
                ]));
            } else {
                lines.push(Line::from(Span::styled(
                    format!("  {}", cl),
                    Style::default().fg(FG),
                )));
            }
        }
    }

    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        chunks[0],
    );

    // Input area — full border box
    let input_idx = 1;
    let is_focused = app.focused_panel == Panel::Chat;
    let border_c = if is_focused { ACCENT } else { MUTED };
    let input_bg = if is_focused {
        Color::Rgb(35, 40, 60)
    } else {
        SURFACE
    };

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_c))
        .style(Style::default().bg(input_bg));

    let input_inner = input_block.inner(chunks[input_idx]);
    frame.render_widget(input_block, chunks[input_idx]);

    let input_text = if app.is_searching {
        Text::from(Line::from(vec![
            Span::styled(
                " search/",
                Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
            ),
            Span::styled(&app.search_query, Style::default().fg(FG)),
            Span::styled("_", Style::default().fg(input_bg).bg(YELLOW)),
        ]))
    } else if app.input.is_empty() {
        Text::from(Line::from(vec![
            Span::styled(
                " > ",
                Style::default().fg(border_c).add_modifier(Modifier::BOLD),
            ),
            Span::styled("type a goal and press Enter...", Style::default().fg(MUTED)),
        ]))
    } else {
        let cursor = app.cursor.min(app.input.len());
        let cc = app.input[cursor..].chars().next().unwrap_or(' ');
        let before = &app.input[..cursor];
        let after = &app.input[(cursor + cc.len_utf8()).min(app.input.len())..];
        Text::from(Line::from(vec![
            Span::styled(
                " > ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(before, Style::default().fg(FG)),
            Span::styled(
                cc.to_string(),
                Style::default()
                    .fg(input_bg)
                    .bg(FG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(after, Style::default().fg(FG)),
        ]))
    };

    frame.render_widget(
        Paragraph::new(input_text).style(Style::default().bg(input_bg)),
        input_inner,
    );
}

// ── Side Panel (tabbed) ──────────────────────────────────────────────

fn render_side_panel(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(0)])
        .split(area);

    // Numbered tab bar
    let tabs_str: Vec<String> = Panel::ALL
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let num = i + 1;
            if p == &app.focused_panel {
                format!("[{} {}]", num, p.name())
            } else {
                format!(" {} {} ", num, p.name())
            }
        })
        .collect();

    let tab_spans: Vec<Span> = tabs_str
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let is_sel = Panel::ALL[i] == app.focused_panel;
            let style = if is_sel {
                Style::default()
                    .fg(BG)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(MUTED).bg(SURFACE)
            };
            Span::styled(s.clone(), style)
        })
        .collect();

    frame.render_widget(
        Paragraph::new(Line::from(tab_spans)).style(Style::default().bg(SURFACE)),
        chunks[0],
    );

    // Content
    let inner = chunks[1];
    match app.focused_panel {
        Panel::Agents => render_agents_view(frame, inner, app),
        Panel::TaskGraph => render_graph_view(frame, inner, app),
        Panel::Locks => render_locks_view(frame, inner, app),
        Panel::Tools => render_tools_view(frame, inner, app),
        Panel::Logs => render_logs_view(frame, inner, app),
        Panel::History => render_history_view(frame, inner, app),
        Panel::Conflicts => render_conflicts_view(frame, inner, app),
        _ => render_agents_view(frame, inner, app),
    }
}

fn side_block(title: &str) -> Block<'_> {
    Block::default()
        .title(format!(" {} ", title))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(MUTED))
        .style(Style::default().bg(BG))
}

fn empty(msg: &str) -> Paragraph<'_> {
    Paragraph::new(Text::from(Span::styled(msg, Style::default().fg(MUTED))))
}

// ── Agents View (main side panel view) ──────────────────────────────

fn render_agents_view(frame: &mut Frame, area: Rect, app: &App) {
    let block = side_block("Agents");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let active_run = app
        .active_run_id
        .as_deref()
        .or_else(|| app.running_runs.first().map(String::as_str))
        .map(|id| id.chars().take(8).collect::<String>())
        .unwrap_or_else(|| "-".into());

    let mut items: Vec<ListItem> = vec![
        ListItem::new(Line::from(vec![
            Span::styled("run:", Style::default().fg(MUTED)),
            Span::styled(
                active_run,
                Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
            ),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("agents:", Style::default().fg(MUTED)),
            Span::styled(app.orch.agents.len().to_string(), Style::default().fg(FG)),
            Span::styled(" locks:", Style::default().fg(MUTED)),
            Span::styled(
                app.orch.locks.len().to_string(),
                Style::default().fg(YELLOW),
            ),
            Span::styled(" queued:", Style::default().fg(MUTED)),
            Span::styled(
                app.orch.write_queue.len().to_string(),
                Style::default().fg(MAGENTA),
            ),
        ])),
        ListItem::new(Line::from("")),
    ];

    if app.orch.agents.is_empty() {
        items.push(ListItem::new(Line::from(Span::styled(
            "No active agents. Submit a goal to begin.",
            Style::default().fg(MUTED),
        ))));
        frame.render_widget(List::new(items), inner);
        return;
    }

    items.extend(app.orch.agents.iter().flat_map(|a| {
        let (icon, color) = phase_icon(&a.phase);
        let pct = a.progress_pct;
        let bar = make_progress_bar(pct, 8);
        let lock = if a.held_locks.is_empty() {
            "-".to_string()
        } else {
            a.held_locks.join(",")
        };
        let queue = if a.queued_write_locks.is_empty() {
            "-".to_string()
        } else {
            a.queued_write_locks.join(",")
        };
        let reads = if a.read_files.is_empty() {
            "-".to_string()
        } else {
            a.read_files.join(",")
        };
        let writes = if a.write_files.is_empty() {
            "-".to_string()
        } else {
            a.write_files.join(",")
        };
        let tool = a.current_tool_call.as_deref().unwrap_or("-");
        let err = a.last_error.as_deref().unwrap_or("");
        let event_age = a
            .last_event_age_secs
            .map(|age| format!(" event:{}s", age))
            .unwrap_or_default();
        let blocked = a
            .blocked_reason
            .as_deref()
            .map(|reason| format!("  blocked: {}", truncate(reason, 48)));
        let heartbeat = if matches!(
            a.phase.as_str(),
            "running"
                | "planning"
                | "decomposing"
                | "spawning_agents"
                | "waiting_for_model"
                | "running_tool"
                | "waiting_for_lock"
                | "approval_needed"
                | "retrying"
        ) {
            a.heartbeat.as_str()
        } else {
            " "
        };

        let mut rows = vec![
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{}{} ", icon, heartbeat),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    &a.label,
                    Style::default().fg(FG).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {}% {}", pct, bar),
                    Style::default().fg(if pct == 100 { GREEN } else { MUTED }),
                ),
            ])),
            ListItem::new(Line::from(Span::styled(
                format!("  task: {}", truncate(&a.current_task, 54)),
                Style::default().fg(FG),
            ))),
            ListItem::new(Line::from(Span::styled(
                format!(
                    "  stage:{} call:{} elapsed:{}s{}",
                    a.stage, tool, a.elapsed_secs, event_age
                ),
                Style::default().fg(MUTED),
            ))),
            ListItem::new(Line::from(Span::styled(
                format!("  retries:{} calls:{}", a.retry_count, a.tool_calls),
                Style::default().fg(MUTED),
            ))),
            ListItem::new(Line::from(Span::styled(
                format!(
                    "  read:{} write:{}",
                    truncate(&reads, 24),
                    truncate(&writes, 24)
                ),
                Style::default().fg(CYAN),
            ))),
            ListItem::new(Line::from(Span::styled(
                format!(
                    "  held:{} queued:{}",
                    truncate(&lock, 24),
                    truncate(&queue, 24)
                ),
                Style::default().fg(YELLOW),
            ))),
            ListItem::new(Line::from(Span::styled(
                format!(
                    "  last:{}{}",
                    truncate(&a.last_update, 42),
                    if err.is_empty() {
                        String::new()
                    } else {
                        format!(" err:{}", truncate(err, 20))
                    }
                ),
                Style::default().fg(if err.is_empty() { MUTED } else { RED }),
            ))),
        ];
        if let Some(blocked) = blocked {
            rows.push(ListItem::new(Line::from(Span::styled(
                blocked,
                Style::default().fg(YELLOW),
            ))));
        }
        rows.push(ListItem::new(Line::from("")));
        rows
    }));

    frame.render_widget(List::new(items), inner);
}

pub fn make_progress_bar(pct: u32, width: usize) -> String {
    let filled = (pct as usize * width / 100).min(width);
    format!("[{}{}]", "=".repeat(filled), " ".repeat(width - filled))
}

fn truncate(value: &str, width: usize) -> String {
    let mut chars = value.chars();
    let mut out: String = chars.by_ref().take(width).collect();
    if chars.next().is_some() && width > 3 {
        let keep = width.saturating_sub(3);
        out = value.chars().take(keep).collect();
        out.push_str("...");
    }
    out
}

// ── Task Graph View ─────────────────────────────────────────────────

fn render_graph_view(frame: &mut Frame, area: Rect, app: &App) {
    let block = side_block("Task Graph");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.orch.task_graph_nodes.is_empty() {
        frame.render_widget(empty("No graph data yet."), inner);
        return;
    }

    let items: Vec<ListItem> = app
        .orch
        .task_graph_nodes
        .iter()
        .map(|n| {
            let indent = "  ".repeat(n.depth);
            let (icon, color) = phase_icon(&n.status);
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{}{} ", indent, if n.depth == 0 { "*" } else { "+-" }),
                    Style::default().fg(MUTED),
                ),
                Span::styled(
                    icon,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ", Style::default()),
                Span::styled(
                    &n.label,
                    Style::default().fg(FG).add_modifier(Modifier::BOLD),
                ),
            ]))
        })
        .collect();

    frame.render_widget(List::new(items), inner);
}

// ── Locks View ──────────────────────────────────────────────────────

fn render_locks_view(frame: &mut Frame, area: Rect, app: &App) {
    let block = side_block("File Locks");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.orch.locks.is_empty() && app.orch.write_queue.is_empty() {
        frame.render_widget(empty("No locks held or queued."), inner);
        return;
    }

    let mut items: Vec<ListItem> = app
        .orch
        .locks
        .iter()
        .flat_map(|l| {
            let (mark, color) = if l.lock_type == "write" {
                ("W", RED)
            } else {
                ("R", GREEN)
            };
            vec![
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("[{}] ", mark),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(&l.file_path, Style::default().fg(FG)),
                ])),
                ListItem::new(Line::from(Span::styled(
                    format!("  held by {} | queue: {}", l.holder, l.queued_waiters),
                    Style::default().fg(MUTED),
                ))),
            ]
        })
        .collect();

    if !app.orch.write_queue.is_empty() {
        items.push(ListItem::new(Line::from("")));
        items.push(ListItem::new(Line::from(Span::styled(
            "Queued write locks",
            Style::default().fg(MAGENTA).add_modifier(Modifier::BOLD),
        ))));
        items.extend(app.orch.write_queue.iter().map(|q| {
            ListItem::new(Line::from(Span::styled(
                format!(
                    "  {} pos:{} requester:{}",
                    q.file_path,
                    q.position,
                    q.requester.chars().take(8).collect::<String>()
                ),
                Style::default().fg(MUTED),
            )))
        }));
    }

    frame.render_widget(List::new(items), inner);
}

// ── Tools View ──────────────────────────────────────────────────────

fn render_tools_view(frame: &mut Frame, area: Rect, app: &App) {
    let block = side_block("Tools");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.tool_displays.is_empty() {
        frame.render_widget(empty("No tool data loaded."), inner);
        return;
    }

    let mut items: Vec<ListItem> = Vec::new();
    if app
        .tool_displays
        .iter()
        .all(|tool| tool.reliability.is_none())
    {
        items.push(ListItem::new(Line::from(Span::styled(
            "No persisted reliability samples.",
            Style::default().fg(MUTED),
        ))));
    }
    items.extend(app.tool_displays.iter().map(|t| {
        let (mark, color) = if t.is_dangerous {
            ("[!]", RED)
        } else {
            ("[v]", GREEN)
        };
        let rel = t
            .reliability
            .as_ref()
            .map(|r| {
                format!(
                    " {:.0}% n={} s{} p{} v{} x{} t{} {:.0}ms",
                    r.score * 100.0,
                    r.total_calls,
                    r.success_count,
                    r.policy_denial_count,
                    r.validation_failure_count,
                    r.transport_failure_count,
                    r.tool_failure_count,
                    r.avg_duration_ms,
                )
            })
            .unwrap_or_default();
        ListItem::new(Line::from(vec![
            Span::styled(
                format!("{} ", mark),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                &t.name,
                Style::default().fg(FG).add_modifier(Modifier::BOLD),
            ),
            Span::styled(rel, Style::default().fg(MUTED)),
        ]))
    }));

    frame.render_widget(List::new(items), inner);
}

// ── Logs/Audit View ─────────────────────────────────────────────────

fn render_logs_view(frame: &mut Frame, area: Rect, app: &App) {
    let title = if app.is_searching {
        format!("Logs /{}", app.search_query)
    } else {
        "Logs".into()
    };
    let block = side_block(&title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.log_entries.is_empty() {
        frame.render_widget(empty("No log entries."), inner);
        return;
    }

    let entries: Vec<&app::LogEntry> = if !app.search_query.is_empty() {
        let q = app.search_query.to_lowercase();
        app.log_entries
            .iter()
            .filter(|e| {
                e.message.to_lowercase().contains(&q) || e.source.to_lowercase().contains(&q)
            })
            .collect()
    } else {
        app.log_entries.iter().collect()
    };

    let items: Vec<ListItem> = entries
        .iter()
        .rev()
        .take(inner.height as usize)
        .map(|e| {
            let lc = match e.level {
                LogLevel::Error => RED,
                LogLevel::Warn => YELLOW,
                _ => MUTED,
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:5} ", e.level.name()),
                    Style::default().fg(lc).add_modifier(Modifier::BOLD),
                ),
                Span::styled(&e.message, Style::default().fg(FG)),
            ]))
        })
        .collect();

    frame.render_widget(List::new(items), inner);
}

// ── History View ────────────────────────────────────────────────────

fn render_history_view(frame: &mut Frame, area: Rect, app: &App) {
    let block = side_block("Run History");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.orch.active_runs.is_empty() {
        frame.render_widget(empty("No runs in history."), inner);
        return;
    }

    let items: Vec<ListItem> = app
        .orch
        .active_runs
        .iter()
        .map(|r| {
            let (icon, color) = phase_icon(&r.status);
            let goal = if r.goal.len() > 30 {
                format!("{}...", truncate(&r.goal, 27))
            } else {
                r.goal.clone()
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{} ", icon),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(goal, Style::default().fg(FG)),
                Span::styled(format!(" |{}|", r.status), Style::default().fg(MUTED)),
            ]))
        })
        .collect();

    frame.render_widget(List::new(items), inner);
}

// ── Conflicts View ──────────────────────────────────────────────────

fn render_conflicts_view(frame: &mut Frame, area: Rect, app: &App) {
    let block = side_block("Conflicts");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.orch.conflicts.is_empty() {
        frame.render_widget(empty("No conflicts detected."), inner);
        return;
    }

    let items: Vec<ListItem> = app
        .orch
        .conflicts
        .iter()
        .flat_map(|c| {
            vec![
                ListItem::new(Line::from(vec![
                    Span::styled(
                        "[!] ",
                        Style::default().fg(RED).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        &c.file_path,
                        Style::default().fg(FG).add_modifier(Modifier::BOLD),
                    ),
                ])),
                ListItem::new(Line::from(Span::styled(
                    format!("  {} <> {} both modified", c.agent_a, c.agent_b),
                    Style::default().fg(YELLOW),
                ))),
            ]
        })
        .collect();

    frame.render_widget(List::new(items), inner);
}

// ── Help Modal ──────────────────────────────────────────────────────

fn render_help(frame: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from(Span::styled(
            concat!("Raven Agent  v", env!("CARGO_PKG_VERSION")),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Enter                Submit goal / steer active run",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  Ctrl+D / Esc         Quit",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  Tab / Shift+Tab      Next/prev tab",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  Alt+1..8             Jump to tab",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  Up/Down / PgUp/PgDn  Scroll",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  /                    Search logs",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  /pause /resume       Control active run",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  /cancel              Approval-gated cancel",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  /redirect <text>     Redirect active run",
            Style::default().fg(FG),
        )),
        Line::from(Span::styled(
            "  ?                    Toggle help",
            Style::default().fg(FG),
        )),
        Line::from(""),
        Line::from(Span::styled(
            " Tabs: 1 Chat  2 Agents  3 Graph  4 Locks  5 Tools  6 Logs  7 History  8 Conflicts",
            Style::default().fg(MUTED),
        )),
    ];

    let h = (lines.len() + 4) as u16;
    let w = 52u16;
    let help_area = Rect {
        x: (area.width.saturating_sub(w)) / 2,
        y: (area.height.saturating_sub(h)) / 2,
        width: w.min(area.width),
        height: h.min(area.height),
    };

    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(SURFACE));

    frame.render_widget(Paragraph::new("").style(Style::default().bg(BG)), help_area);
    frame.render_widget(Paragraph::new(Text::from(lines)).block(block), help_area);
}

fn render_approval(frame: &mut Frame, area: Rect, app: &App) {
    let Some(request) = &app.pending_approval else {
        return;
    };
    let lines = vec![
        Line::from(Span::styled(
            &request.title,
            Style::default().fg(RED).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(&request.details, Style::default().fg(FG))),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "[y] approve",
                Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
            ),
            Span::styled("   ", Style::default()),
            Span::styled(
                "[n] deny",
                Style::default().fg(RED).add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let h = 9u16;
    let w = 64u16;
    let modal = Rect {
        x: (area.width.saturating_sub(w)) / 2,
        y: (area.height.saturating_sub(h)) / 2,
        width: w.min(area.width),
        height: h.min(area.height),
    };

    let block = Block::default()
        .title(" Approval Required ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(RED))
        .style(Style::default().bg(SURFACE));

    frame.render_widget(Paragraph::new("").style(Style::default().bg(BG)), modal);
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: false }),
        modal,
    );
}

// ── Main Render ─────────────────────────────────────────────────────

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let width = area.width as usize;

    // Responsive: keep the live side panel visible on standard 80-column SSH
    // terminals, but collapse it on very narrow terminals where two panes
    // would make both unreadable.
    let side_pct = if width >= 120 {
        33
    } else if width >= 80 {
        35
    } else if width >= 70 {
        40
    } else {
        0
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(4),
            Constraint::Length(1),
        ])
        .split(area);

    render_top_bar(frame, chunks[0], app);

    if side_pct > 0 {
        let main = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(100 - side_pct),
                Constraint::Percentage(side_pct),
            ])
            .split(chunks[1]);
        render_chat_panel(frame, main[0], app);
        render_side_panel(frame, main[1], app);
    } else {
        render_chat_panel(frame, chunks[1], app);
    }

    render_bottom_bar(frame, chunks[2], app);

    // Modals (rendered last, on top)
    if app.show_help {
        render_help(frame, area);
    }
    render_approval(frame, area, app);
}

// ── Event Loop ──────────────────────────────────────────────────────

pub async fn run_ui(app: &mut App) -> anyhow::Result<()> {
    use crossterm::{
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::Terminal;
    use std::io;

    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut handler = crate::events::EventHandler::new(500);

    let result = loop {
        let event = handler.next().await;
        match &event {
            crate::events::Event::Terminal(te) => {
                if let Some(action) = handler.handle_event(te)
                    && let Err(error) = app.handle_action(action).await
                {
                    app.on_run_failed("", &error.to_string());
                }
            }
            crate::events::Event::Tick => {
                app.dispatch(crate::events::Action::Tick);
                app.drain_runner_events();
                app.refresh_orchestration().await.ok();
            }
        }
        app.drain_runner_events();
        let _ = terminal.draw(|frame| render(frame, app));
        if app.should_quit {
            break Ok(());
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}
