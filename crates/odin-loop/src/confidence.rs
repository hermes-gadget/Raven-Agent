//! Confidence scoring for model outputs.
//!
//! Evaluates how well a model's action achieved its intended goal.
//! Used by the Critique phase to decide whether to continue, retry, or escalate.

use odin_core::types::ConfidenceScore;

/// Scores confidence in an agent's action.
///
/// For smaller models, confidence scoring uses simple heuristics
/// rather than an additional model call (saving tokens and latency).
/// A stronger model can optionally be used for more accurate scoring.
#[derive(Debug, Clone)]
pub struct ConfidenceScorer {
    /// Whether to use a model for scoring (vs heuristics)
    pub use_model: bool,
    /// Threshold below which to flag for revision
    pub low_threshold: f64,
    /// Threshold above which to consider high confidence
    pub high_threshold: f64,
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self {
            use_model: false,
            low_threshold: 0.5,
            high_threshold: 0.8,
        }
    }
}

impl ConfidenceScorer {
    /// Score a tool execution result.
    ///
    /// Heuristic scoring considers:
    /// - Did the tool succeed? (+0.5 base if yes, 0.0 if no)
    /// - Did it produce output? (+0.2 if non-empty output)
    /// - Was there an error? (-0.3 if error present)
    /// - Did it run fast? (+0.1 if under 1 second)
    pub fn score_tool_result(
        &self,
        success: bool,
        has_output: bool,
        has_error: bool,
        duration_ms: u64,
    ) -> ConfidenceScore {
        let mut score = 0.0;

        // Base: success or failure
        if success {
            score += 0.5;
        }

        // Output present = more confidence
        if has_output {
            score += 0.2;
        }

        // Error present = less confidence
        if has_error {
            score -= 0.3;
        }

        // Fast execution = more confidence (no timeout/hanging)
        if duration_ms < 1000 {
            score += 0.1;
        } else if duration_ms > 10000 {
            score -= 0.1;
        }

        ConfidenceScore::new(score)
    }

    /// Score a text response from the model.
    ///
    /// Heuristic scoring considers:
    /// - Empty or clearly truncated responses
    /// - Presence of error indicators ("error", "failed", "sorry")
    /// - Presence of confidence indicators ("I'm confident", "definitely")
    /// - Whether it addresses the goal
    pub fn score_text_response(&self, response: &str, goal_hint: Option<&str>) -> ConfidenceScore {
        let mut score = 0.5; // Start at neutral

        let lower = response.to_lowercase();

        // Empty/near-empty output is incomplete. Length never increases
        // confidence: verbose text is not evidence of successful work.
        if response.trim().len() < 4 {
            score -= 0.3;
        }

        // Truncation indicators
        if lower.contains("truncated") || lower.contains("cut off") {
            score -= 0.2;
        }

        // Error indicators
        let error_words = [
            "error",
            "failed",
            "unable",
            "cannot",
            "sorry",
            "apologize",
            "unfortunately",
            "not able",
            "doesn't work",
            "not possible",
        ];
        let error_count = error_words.iter().filter(|w| lower.contains(*w)).count();
        if error_count > 0 {
            score -= 0.1 * (error_count.min(3) as f64);
        }

        // Confidence indicators
        let confidence_words = [
            "confident",
            "definitely",
            "certainly",
            "clearly",
            "successfully",
            "correct",
            "verified",
        ];
        let conf_count = confidence_words
            .iter()
            .filter(|w| lower.contains(*w))
            .count();
        if conf_count > 0 {
            score += 0.05 * (conf_count.min(4) as f64);
        }

        // Goal alignment (simple keyword check)
        if let Some(goal) = goal_hint {
            let goal_lower = goal.to_lowercase();
            let goal_words: Vec<&str> = goal_lower.split_whitespace().collect();
            let matched = goal_words.iter().filter(|w| lower.contains(*w)).count();
            if matched > 0 {
                score += 0.05 * (matched.min(4) as f64);
            }
        }

        ConfidenceScore::new(score)
    }

    /// Check if the score indicates escalation is needed.
    pub fn should_escalate(&self, score: ConfidenceScore, retry_count: u32) -> bool {
        score.value() < self.low_threshold && retry_count >= 2
    }

    /// Check if the score is acceptable to continue.
    pub fn is_acceptable(&self, score: ConfidenceScore) -> bool {
        score.value() >= self.low_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_tool_success() {
        let scorer = ConfidenceScorer::default();
        let score = scorer.score_tool_result(true, true, false, 500);
        assert!(score.value() > 0.7);
    }

    #[test]
    fn test_score_tool_failure() {
        let scorer = ConfidenceScorer::default();
        let score = scorer.score_tool_result(false, false, true, 5000);
        assert!(score.value() < 0.3);
    }

    #[test]
    fn test_score_text_error() {
        let scorer = ConfidenceScorer::default();
        let score = scorer.score_text_response(
            "I'm sorry, I failed to do that. Unfortunately there was an error.",
            Some("complete the task"),
        );
        assert!(score.value() < 0.5);
    }

    #[test]
    fn test_score_text_confident() {
        let scorer = ConfidenceScorer::default();
        let score = scorer.score_text_response(
            "I've successfully completed the task. The file was written correctly and verified.",
            Some("write the file"),
        );
        assert!(score.value() > 0.5);
    }

    #[test]
    fn test_confidence_clamped() {
        let score = ConfidenceScore::new(2.0);
        assert_eq!(score.value(), 1.0);
        let score = ConfidenceScore::new(-1.0);
        assert_eq!(score.value(), 0.0);
    }
}
