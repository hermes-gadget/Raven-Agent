//! Task graph: parent goal → sub-goals → agents → files/tools → outputs.
//!
//! A directed acyclic graph (DAG) representing the decomposition of a user's goal
//! into executable work units. Each node is a sub-goal that can be assigned to a
//! sub-agent. Edges represent dependencies (must complete before downstream work).
//!
//! The graph is validated topologically — cycles are rejected, and independent
//! sub-trees can run in parallel.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use uuid::Uuid;

/// Unique identifier for a task graph node.
pub type NodeId = Uuid;

/// A node in the task graph — one sub-goal for a sub-agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Human-readable label (e.g., "fix-cli-bug", "improve-docs").
    pub label: String,
    /// The goal text the sub-agent should execute.
    pub goal: String,
    /// Files this node needs read access to.
    pub read_files: Vec<String>,
    /// Files this node will write/modify.
    pub write_files: Vec<String>,
    /// Capability tags required (e.g., ["shell", "git", "web"]).
    pub required_capabilities: Vec<String>,
    /// Estimated priority (lower = higher priority).
    pub priority: u32,
    /// Current status.
    pub status: TaskNodeStatus,
    /// Result summary, if completed.
    pub result: Option<String>,
    /// Assigned sub-agent ID, if spawned.
    pub agent_id: Option<Uuid>,
}

/// Status of a task graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskNodeStatus {
    /// Ready to be scheduled.
    Pending,
    /// Waiting for upstream dependencies to complete.
    Blocked,
    /// Assigned to a sub-agent and running.
    Running,
    /// Completed successfully.
    Done,
    /// Failed with an error.
    Failed,
    /// Cancelled by user or orchestrator.
    Cancelled,
}

/// An edge in the task graph — dependency from `from` to `to`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEdge {
    /// Source node (dependency).
    pub from: NodeId,
    /// Target node (depends on source).
    pub to: NodeId,
    /// Optional label (e.g., "depends-on", "after").
    pub label: Option<String>,
}

/// The task graph — a DAG of work units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraph {
    /// Stable identifier used by persistence and external APIs.
    #[serde(default = "default_graph_id")]
    pub id: Uuid,
    /// All nodes in the graph.
    pub nodes: HashMap<NodeId, TaskNode>,
    /// All edges (dependencies).
    pub edges: Vec<TaskEdge>,
    /// The root goal / user intent.
    pub root_goal: String,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Overall status.
    pub status: TaskGraphStatus,
}

/// Overall task graph status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskGraphStatus {
    /// Graph is being built / decomposed.
    Building,
    /// Execution in progress.
    Running,
    /// Execution is intentionally paused.
    Paused,
    /// All nodes complete.
    Complete,
    /// Some nodes failed, graph aborted.
    Failed,
    /// Cancelled by user.
    Cancelled,
}

impl TaskGraph {
    /// Create a new empty task graph for a root goal.
    pub fn new(root_goal: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            nodes: HashMap::new(),
            edges: Vec::new(),
            root_goal: root_goal.into(),
            created_at: chrono::Utc::now(),
            status: TaskGraphStatus::Building,
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: TaskNode) -> NodeId {
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }

    /// Add an edge (dependency) to the graph.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> &mut Self {
        self.edges.push(TaskEdge {
            from,
            to,
            label: None,
        });
        self
    }

    /// Add a labeled edge.
    pub fn add_labeled_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        label: impl Into<String>,
    ) -> &mut Self {
        self.edges.push(TaskEdge {
            from,
            to,
            label: Some(label.into()),
        });
        self
    }

    /// Get all nodes that are ready to execute (no pending upstream dependencies).
    pub fn ready_nodes(&self) -> Vec<&TaskNode> {
        self.nodes
            .values()
            .filter(|n| n.status == TaskNodeStatus::Pending)
            .filter(|n| self.all_upstream_done(n.id))
            .collect()
    }

    /// Check if all upstream dependencies of a node are done.
    fn all_upstream_done(&self, node_id: NodeId) -> bool {
        self.edges.iter().filter(|e| e.to == node_id).all(|e| {
            self.nodes
                .get(&e.from)
                .map(|n| n.status == TaskNodeStatus::Done)
                .unwrap_or(true)
        })
    }

    /// Get all nodes that are independent (no dependencies between them) and can run in parallel.
    pub fn independent_groups(&self) -> Vec<Vec<NodeId>> {
        let mut finished: HashSet<NodeId> = self
            .nodes
            .values()
            .filter(|n| n.status == TaskNodeStatus::Done)
            .map(|n| n.id)
            .collect();

        let mut groups: Vec<Vec<NodeId>> = Vec::new();
        let mut remaining: HashSet<NodeId> = self
            .nodes
            .values()
            .filter(|n| n.status == TaskNodeStatus::Pending)
            .map(|n| n.id)
            .collect();

        while !remaining.is_empty() {
            let group: Vec<NodeId> = remaining
                .iter()
                .filter(|id| {
                    // Node is ready if all upstream are done or finished
                    self.edges
                        .iter()
                        .filter(|e| e.to == **id)
                        .all(|e| finished.contains(&e.from))
                })
                .copied()
                .collect();

            if group.is_empty() {
                // Deadlock — remaining nodes are blocked on each other
                break;
            }

            for id in &group {
                remaining.remove(id);
            }
            finished.extend(group.iter().copied());
            groups.push(group);
        }

        groups
    }

    /// Topological sort. Returns nodes in execution order.
    /// Returns an error if a cycle is detected.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, String> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in self.nodes.keys() {
            in_degree.entry(*node).or_insert(0);
            adjacency.entry(*node).or_default();
        }

        for edge in &self.edges {
            *in_degree.entry(edge.to).or_default() += 1;
            adjacency.entry(edge.from).or_default().push(edge.to);
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted.push(node);

            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if sorted.len() != self.nodes.len() {
            return Err("Cycle detected in task graph".to_string());
        }

        Ok(sorted)
    }

    /// Mark a node's status.
    pub fn update_node_status(&mut self, node_id: NodeId, status: TaskNodeStatus) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.status = status;
        }
    }

    /// Fail pending descendants whose dependencies can no longer succeed.
    ///
    /// The scan repeats so a failed node propagates through dependency chains
    /// in one call. The returned IDs are the nodes transitioned by this call.
    pub fn fail_blocked_descendants(&mut self) -> Vec<NodeId> {
        let mut failed = Vec::new();
        loop {
            let blocked: Vec<_> = self
                .nodes
                .values()
                .filter(|node| {
                    matches!(
                        node.status,
                        TaskNodeStatus::Pending | TaskNodeStatus::Blocked
                    ) && self
                        .edges
                        .iter()
                        .filter(|edge| edge.to == node.id)
                        .any(|edge| {
                            self.nodes.get(&edge.from).is_some_and(|dependency| {
                                matches!(
                                    dependency.status,
                                    TaskNodeStatus::Failed | TaskNodeStatus::Cancelled
                                )
                            })
                        })
                })
                .map(|node| node.id)
                .collect();
            if blocked.is_empty() {
                break;
            }

            for node_id in blocked {
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.status = TaskNodeStatus::Failed;
                    node.result = Some("blocked by a failed or cancelled dependency".into());
                    failed.push(node_id);
                }
            }
        }
        failed
    }

    /// Check if the entire graph is complete.
    pub fn is_complete(&self) -> bool {
        self.nodes
            .values()
            .all(|n| matches!(n.status, TaskNodeStatus::Done | TaskNodeStatus::Cancelled))
    }

    /// Check if any node has failed.
    pub fn has_failures(&self) -> bool {
        self.nodes
            .values()
            .any(|n| n.status == TaskNodeStatus::Failed)
    }

    /// Get progress (completed / total).
    pub fn progress(&self) -> (usize, usize) {
        let total = self.nodes.len();
        let done = self
            .nodes
            .values()
            .filter(|n| n.status == TaskNodeStatus::Done)
            .count();
        (done, total)
    }

    /// Get a summary of the graph state.
    pub fn summary(&self) -> TaskGraphSummary {
        let (done, total) = self.progress();
        let running = self
            .nodes
            .values()
            .filter(|n| n.status == TaskNodeStatus::Running)
            .count();
        let blocked = self
            .nodes
            .values()
            .filter(|n| n.status == TaskNodeStatus::Blocked)
            .count();
        let failed = self
            .nodes
            .values()
            .filter(|n| n.status == TaskNodeStatus::Failed)
            .count();

        TaskGraphSummary {
            root_goal: self.root_goal.clone(),
            status: self.status,
            total_nodes: total,
            done,
            running,
            blocked,
            pending: total - done - running - blocked - failed,
            failed,
        }
    }
}

fn default_graph_id() -> Uuid {
    Uuid::new_v4()
}

/// Summary of a task graph's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraphSummary {
    pub root_goal: String,
    pub status: TaskGraphStatus,
    pub total_nodes: usize,
    pub done: usize,
    pub running: usize,
    pub blocked: usize,
    pub pending: usize,
    pub failed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(label: &str, goal: &str) -> TaskNode {
        TaskNode {
            id: Uuid::new_v4(),
            label: label.into(),
            goal: goal.into(),
            read_files: vec![],
            write_files: vec![],
            required_capabilities: vec![],
            priority: 0,
            status: TaskNodeStatus::Pending,
            result: None,
            agent_id: None,
        }
    }

    #[test]
    fn test_topological_sort_linear() {
        let mut graph = TaskGraph::new("test");
        let a = graph.add_node(make_node("a", "step a"));
        let b = graph.add_node(make_node("b", "step b"));
        let c = graph.add_node(make_node("c", "step c"));
        graph.add_edge(a, b).add_edge(b, c);

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted, vec![a, b, c]);
    }

    #[test]
    fn test_topological_sort_diamond() {
        let mut graph = TaskGraph::new("test");
        let a = graph.add_node(make_node("a", "start"));
        let b = graph.add_node(make_node("b", "left"));
        let c = graph.add_node(make_node("c", "right"));
        let d = graph.add_node(make_node("d", "end"));
        graph
            .add_edge(a, b)
            .add_edge(a, c)
            .add_edge(b, d)
            .add_edge(c, d);

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted[0], a);
        assert_eq!(sorted[3], d);
        // b and c can be in any order
        assert!(sorted[1..3].contains(&b));
        assert!(sorted[1..3].contains(&c));
    }

    #[test]
    fn failed_dependencies_propagate_through_chains_and_diamonds() {
        let mut graph = TaskGraph::new("test");
        let root = graph.add_node(make_node("root", "root"));
        let left = graph.add_node(make_node("left", "left"));
        let right = graph.add_node(make_node("right", "right"));
        let leaf = graph.add_node(make_node("leaf", "leaf"));
        graph
            .add_edge(root, left)
            .add_edge(root, right)
            .add_edge(left, leaf)
            .add_edge(right, leaf);
        graph.update_node_status(root, TaskNodeStatus::Failed);

        let failed = graph.fail_blocked_descendants();

        assert_eq!(failed.len(), 3);
        for node_id in [left, right, leaf] {
            let node = &graph.nodes[&node_id];
            assert_eq!(node.status, TaskNodeStatus::Failed);
            assert!(node.result.as_deref().unwrap().contains("dependency"));
        }
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = TaskGraph::new("test");
        let a = graph.add_node(make_node("a", "a"));
        let b = graph.add_node(make_node("b", "b"));
        let c = graph.add_node(make_node("c", "c"));
        graph.add_edge(a, b).add_edge(b, c).add_edge(c, a); // cycle: a→b→c→a

        assert!(graph.topological_sort().is_err());
    }

    #[test]
    fn test_independent_groups_no_deps() {
        let mut graph = TaskGraph::new("test");
        let _a = graph.add_node(make_node("a", "a"));
        let _b = graph.add_node(make_node("b", "b"));
        let _c = graph.add_node(make_node("c", "c"));

        let groups = graph.independent_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn test_independent_groups_with_deps() {
        let mut graph = TaskGraph::new("test");
        let a = graph.add_node(make_node("a", "a"));
        let b = graph.add_node(make_node("b", "b"));
        let c = graph.add_node(make_node("c", "c"));
        graph.add_edge(a, b); // b depends on a

        // Mark 'a' as done, so 'b' is the only ready node
        graph.update_node_status(a, TaskNodeStatus::Done);

        let groups = graph.independent_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2); // both c and b are ready
        assert!(groups[0].contains(&b));
        assert!(groups[0].contains(&c));
    }

    #[test]
    fn test_independent_groups_progress_through_chain() {
        let mut graph = TaskGraph::new("chain");
        let a = graph.add_node(make_node("a", "a"));
        let b = graph.add_node(make_node("b", "b"));
        let c = graph.add_node(make_node("c", "c"));
        graph.add_edge(a, b).add_edge(b, c);

        let groups = graph.independent_groups();
        assert_eq!(groups, vec![vec![a], vec![b], vec![c]]);
    }

    #[test]
    fn test_ready_nodes() {
        let mut graph = TaskGraph::new("test");
        let a = graph.add_node(make_node("a", "a"));
        let b = graph.add_node(make_node("b", "b"));
        graph.add_edge(a, b);

        // a is ready (no deps), b is blocked (needs a)
        let ready = graph.ready_nodes();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, a);
    }

    #[test]
    fn test_progress_tracking() {
        let mut graph = TaskGraph::new("test");
        let a = graph.add_node(make_node("a", "a"));
        let b = graph.add_node(make_node("b", "b"));
        graph.update_node_status(a, TaskNodeStatus::Done);

        let (done, total) = graph.progress();
        assert_eq!(done, 1);
        assert_eq!(total, 2);
        assert!(!graph.is_complete());

        graph.update_node_status(b, TaskNodeStatus::Done);
        assert!(graph.is_complete());
    }

    #[test]
    fn test_summary() {
        let mut graph = TaskGraph::new("test root");
        let a = graph.add_node(make_node("a", "a"));
        let b = graph.add_node(make_node("b", "b"));
        graph.update_node_status(a, TaskNodeStatus::Done);
        graph.update_node_status(b, TaskNodeStatus::Running);

        let summary = graph.summary();
        assert_eq!(summary.total_nodes, 2);
        assert_eq!(summary.done, 1);
        assert_eq!(summary.running, 1);
        assert_eq!(summary.root_goal, "test root");
    }
}
