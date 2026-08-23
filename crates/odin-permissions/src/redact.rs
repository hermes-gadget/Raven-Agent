//! Secret and PII redactor — detects and redacts API keys, tokens, secrets, and PII.
//!
//! ## Design
//!
//! Two-layer redaction with configurable levels:
//! - **Secrets**: API keys, tokens, private keys, passwords, connection strings
//! - **PII**: emails, phone numbers, SSNs, credit cards, IP addresses
//!
//! Each layer can be enabled/disabled independently. The `RedactionLevel` enum
//! provides presets: `SecretsOnly`, `PIIOnly`, `Full`, `Custom`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use odin_permissions::redact::SecretRedactor;
//!
//! let redactor = SecretRedactor::full(); // secrets + PII
//! let safe = redactor.redact("API_KEY=sk-abc123... email: user@example.com");
//! // → "API_KEY=[REDACTED:OpenAI API key] email: [REDACTED:Email address]"
//! ```

use regex::Regex;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls which categories of sensitive data to redact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedactionLevel {
    /// Only redact secrets (API keys, tokens, private keys, passwords).
    SecretsOnly,
    /// Only redact PII (emails, phone numbers, SSNs, credit cards, IPs).
    PiiOnly,
    /// Redact both secrets and PII (default).
    Full,
    /// User-provided mask list via `RedactionConfig`.
    Custom,
}

/// Fine-grained control over which patterns are active.
#[derive(Debug, Clone)]
pub struct RedactionConfig {
    /// Redaction level preset.
    pub level: RedactionLevel,
    /// Enable secret patterns. Default: true.
    pub enable_secrets: bool,
    /// Enable PII patterns. Default: true.
    pub enable_pii: bool,
    /// Additional custom regex patterns.
    pub custom_patterns: Vec<(Regex, String)>,
    /// Patterns to exclude (by type name).
    pub exclude_patterns: Vec<String>,
}

impl Default for RedactionConfig {
    fn default() -> Self {
        Self {
            level: RedactionLevel::Full,
            enable_secrets: true,
            enable_pii: true,
            custom_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
        }
    }
}

impl RedactionConfig {
    /// Secrets-only preset.
    pub fn secrets_only() -> Self {
        Self {
            level: RedactionLevel::SecretsOnly,
            enable_secrets: true,
            enable_pii: false,
            ..Default::default()
        }
    }

    /// PII-only preset.
    pub fn pii_only() -> Self {
        Self {
            level: RedactionLevel::PiiOnly,
            enable_secrets: false,
            enable_pii: true,
            ..Default::default()
        }
    }

    /// Full redaction (secrets + PII).
    pub fn full() -> Self {
        Self {
            level: RedactionLevel::Full,
            enable_secrets: true,
            enable_pii: true,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern helpers
// ---------------------------------------------------------------------------

/// A compiled detection pattern with its human-readable name.
#[derive(Clone)]
struct Pattern {
    regex: Regex,
    name: String,
    category: PatternCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternCategory {
    Secret,
    Pii,
}

fn compile(pattern: &str, name: &str, category: PatternCategory) -> Pattern {
    Pattern {
        regex: Regex::new(pattern).expect("invalid regex in SecretRedactor"),
        name: name.to_string(),
        category,
    }
}

// ---------------------------------------------------------------------------
// SecretRedactor
// ---------------------------------------------------------------------------

/// A compiled set of patterns for detecting secrets and PII in text.
#[derive(Clone)]
pub struct SecretRedactor {
    patterns: Vec<Pattern>,
    config: RedactionConfig,
}

impl SecretRedactor {
    // -- constructors -------------------------------------------------------

    /// Create a redactor with the given configuration.
    pub fn with_config(config: RedactionConfig) -> Self {
        let all_patterns = Self::build_all_patterns();
        let mut filtered: Vec<Pattern> = all_patterns
            .into_iter()
            .filter(|p| match p.category {
                PatternCategory::Secret => config.enable_secrets,
                PatternCategory::Pii => config.enable_pii,
            })
            .filter(|p| !config.exclude_patterns.contains(&p.name))
            .collect();

        // Append any custom user-supplied patterns.
        for (regex, name) in &config.custom_patterns {
            filtered.push(Pattern {
                regex: regex.clone(),
                name: name.clone(),
                category: PatternCategory::Secret, // custom patterns are secrets by default
            });
        }

        Self {
            patterns: filtered,
            config,
        }
    }

    /// Full redaction (secrets + PII) — the default.
    pub fn full() -> Self {
        Self::with_config(RedactionConfig::full())
    }

    /// Secrets-only redaction.
    pub fn secrets_only() -> Self {
        Self::with_config(RedactionConfig::secrets_only())
    }

    /// PII-only redaction.
    pub fn pii_only() -> Self {
        Self::with_config(RedactionConfig::pii_only())
    }

    /// Create a new redactor with the default full set of patterns.
    /// Kept for backward compatibility.
    pub fn new() -> Self {
        Self::full()
    }

    // -- pattern construction -----------------------------------------------

    fn build_all_patterns() -> Vec<Pattern> {
        vec![
            // ── Secrets: API Keys & Tokens ──────────────────────────────────
            compile(
                r"sk-[a-zA-Z0-9]{20,}",
                "OpenAI API key",
                PatternCategory::Secret,
            ),
            compile(
                r"sk-ant-[a-zA-Z0-9_-]{20,}",
                "Anthropic API key",
                PatternCategory::Secret,
            ),
            compile(
                r"gh[pousr]_[a-zA-Z0-9]{20,}",
                "GitHub token",
                PatternCategory::Secret,
            ),
            compile(
                r"glpat-[a-zA-Z0-9_-]{20,}",
                "GitLab personal access token",
                PatternCategory::Secret,
            ),
            compile(
                r"xox[bprs]-\d{11,}-\d{11,}-[a-zA-Z0-9]{24,}",
                "Slack token",
                PatternCategory::Secret,
            ),
            compile(
                r"AKIA[0-9A-Z]{16}",
                "AWS access key",
                PatternCategory::Secret,
            ),
            compile(
                r"(?i)aws_secret_access_key[=:]\s*[a-zA-Z0-9/+=]{20,}",
                "AWS secret key",
                PatternCategory::Secret,
            ),
            compile(
                r"(?i)(api[_-]?key|apikey|api[_-]?secret)[=:]\s*[a-zA-Z0-9_-]{16,}",
                "Generic API key assignment",
                PatternCategory::Secret,
            ),
            compile(
                r"(?i)(token|secret|password|passwd)[=:]\s*\S{8,}",
                "Credential assignment",
                PatternCategory::Secret,
            ),
            compile(
                r"(?i)(Authorization|auth)[=:]\s*[Bb]earer\s+[a-zA-Z0-9._\-+/=]{20,}",
                "Authorization header",
                PatternCategory::Secret,
            ),
            compile(
                r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}",
                "JWT token",
                PatternCategory::Secret,
            ),
            // ── Secrets: Private Keys ───────────────────────────────────────
            compile(
                r"(?s)-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----",
                "Private key block",
                PatternCategory::Secret,
            ),
            compile(
                r"(?s)-----BEGIN PGP PRIVATE KEY BLOCK-----.*?-----END PGP PRIVATE KEY BLOCK-----",
                "PGP private key block",
                PatternCategory::Secret,
            ),
            compile(
                r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----",
                "Private key header",
                PatternCategory::Secret,
            ),
            compile(
                r"-----BEGIN PGP PRIVATE KEY BLOCK-----",
                "PGP private key",
                PatternCategory::Secret,
            ),
            // ── Secrets: Connection Strings ──────────────────────────────────
            compile(
                r"(?i)(?:mongodb|mysql|postgres(?:ql)?|redis|sqlite)://[^@\s]+:[^@\s]+@[^\s]+",
                "Database connection string",
                PatternCategory::Secret,
            ),
            compile(
                r"(?i)connection[_-]?string[=:]\s*\S{10,}",
                "Connection string assignment",
                PatternCategory::Secret,
            ),
            // ── Secrets: Stripe / Payment ────────────────────────────────────
            compile(
                r"(?:sk|pk)_(?:live|test)_[a-zA-Z0-9]{24,}",
                "Stripe key",
                PatternCategory::Secret,
            ),
            // ── Secrets: Bearer tokens (standalone) ──────────────────────────
            compile(
                r"(?i)bearer\s+([a-zA-Z0-9._\-+/=]{20,})",
                "Bearer token",
                PatternCategory::Secret,
            ),
            // ── PII: Email addresses ────────────────────────────────────────
            compile(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "Email address",
                PatternCategory::Pii,
            ),
            // ── PII: Phone numbers (US + international variants) ────────────
            // Require at least one separator (dash, dot, space, or parens) to avoid
            // false positives on SHA hashes, serial numbers, etc.
            compile(
                r"(?:\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}",
                "Phone number",
                PatternCategory::Pii,
            ),
            // ── PII: SSN (US) ───────────────────────────────────────────────
            compile(r"\b\d{3}-\d{2}-\d{4}\b", "SSN", PatternCategory::Pii),
            // ── PII: Credit card numbers ────────────────────────────────────
            compile(
                r"\b(?:\d[ -]*?){13,16}\b",
                "Credit card number",
                PatternCategory::Pii,
            ),
            // ── PII: IPv4 & IPv6 addresses ──────────────────────────────────
            compile(
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                "IPv4 address",
                PatternCategory::Pii,
            ),
            compile(
                r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
                "IPv6 address",
                PatternCategory::Pii,
            ),
            // ── PII: Long numeric sequences (potential account numbers) ─────
            // 16+ digits to avoid false positives on order #s, serial #s, etc.
            // and to not overlap with credit card detection (13-16 digits).
            compile(r"\b\d{16,}\b", "Numeric identifier", PatternCategory::Pii),
        ]
    }

    // -- public API --------------------------------------------------------

    /// Redact secrets and/or PII from a string, replacing them with `[REDACTED:<type>]`.
    ///
    /// The patterns are applied in order; earlier patterns take precedence.
    pub fn redact(&self, input: &str) -> String {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(input)
            && matches!(
                value,
                serde_json::Value::Object(_) | serde_json::Value::Array(_)
            )
        {
            let redacted = self.redact_json_value(&value);
            return if redacted == value {
                input.to_string()
            } else {
                redacted.to_string()
            };
        }
        self.redact_text(input)
    }

    fn redact_text(&self, input: &str) -> String {
        let mut result = input.to_string();
        for pattern in &self.patterns {
            // Skip credit-card check for strings that are clearly not numbers
            if pattern.name == "Credit card number" && !Self::could_be_credit_card(input) {
                continue;
            }
            result = pattern
                .regex
                .replace_all(&result, format!("[REDACTED:{}]", pattern.name))
                .to_string();
        }
        result
    }

    /// Redact a `serde_json::Value` recursively — walks objects, arrays, and strings.
    pub fn redact_json(&self, value: &serde_json::Value) -> serde_json::Value {
        self.redact_json_value(value)
    }

    fn redact_json_value(&self, value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::String(s) => serde_json::Value::String(self.redact_text(s)),
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| self.redact_json_value(v)).collect())
            }
            serde_json::Value::Object(obj) => {
                let mut new_obj = serde_json::Map::new();
                for (k, v) in obj {
                    let value = if self.config.enable_secrets && Self::is_sensitive_json_key(k) {
                        serde_json::Value::String("[REDACTED:Sensitive JSON field]".into())
                    } else {
                        self.redact_json_value(v)
                    };
                    new_obj.insert(k.clone(), value);
                }
                serde_json::Value::Object(new_obj)
            }
            other => other.clone(),
        }
    }

    /// Check if a string contains any detectable secrets or PII.
    pub fn contains_secrets(&self, input: &str) -> bool {
        self.patterns.iter().any(|p| p.regex.is_match(input))
    }

    /// List the types of sensitive data found in a string.
    pub fn detect(&self, input: &str) -> Vec<String> {
        self.patterns
            .iter()
            .filter(|p| p.regex.is_match(input))
            .map(|p| p.name.clone())
            .collect()
    }

    /// List the types of secrets found in a string (excludes PII).
    pub fn detect_secrets(&self, input: &str) -> Vec<String> {
        self.patterns
            .iter()
            .filter(|p| p.category == PatternCategory::Secret && p.regex.is_match(input))
            .map(|p| p.name.clone())
            .collect()
    }

    /// List the types of PII found in a string (excludes secrets).
    pub fn detect_pii(&self, input: &str) -> Vec<String> {
        self.patterns
            .iter()
            .filter(|p| p.category == PatternCategory::Pii && p.regex.is_match(input))
            .map(|p| p.name.clone())
            .collect()
    }

    /// Return the count of active patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Return the current configuration.
    pub fn config(&self) -> &RedactionConfig {
        &self.config
    }

    // -- helpers -----------------------------------------------------------

    /// Quick heuristic to avoid running the expensive credit-card regex on
    /// strings that contain very few digits.
    fn could_be_credit_card(input: &str) -> bool {
        let digit_count = input.chars().filter(|c| c.is_ascii_digit()).count();
        digit_count >= 13
    }

    fn is_sensitive_json_key(key: &str) -> bool {
        let normalized: String = key
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        matches!(
            normalized.as_str(),
            "password"
                | "passwd"
                | "token"
                | "accesstoken"
                | "refreshtoken"
                | "secret"
                | "clientsecret"
                | "webhooksecret"
                | "apikey"
                | "apisecret"
                | "authorization"
                | "proxyauthorization"
                | "auth"
                | "privatekey"
                | "credential"
                | "credentials"
        ) || [
            "password",
            "passwd",
            "accesstoken",
            "refreshtoken",
            "clientsecret",
            "webhooksecret",
            "apikey",
            "apisecret",
            "privatekey",
        ]
        .iter()
        .any(|suffix| normalized.ends_with(suffix))
    }
}

impl Default for SecretRedactor {
    fn default() -> Self {
        Self::full()
    }
}

// ---------------------------------------------------------------------------
// Integration utilities
// ---------------------------------------------------------------------------

/// Convenience: redact a string and log what was found.
///
/// Returns `(redacted_text, Vec<detected_types>)`.
pub fn redact_and_detect(input: &str, redactor: &SecretRedactor) -> (String, Vec<String>) {
    let detected = redactor.detect(input);
    let redacted = redactor.redact(input);
    (redacted, detected)
}

/// Check a string and return `None` if clean, `Some(detected_types)` if dirty.
pub fn check_sensitive(input: &str, redactor: &SecretRedactor) -> Option<Vec<String>> {
    let detected = redactor.detect(input);
    if detected.is_empty() {
        None
    } else {
        Some(detected)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers: synthetic test tokens ───────────────────────────────────

    const FAKE_OPENAI: &str = "sk-test0000000000000000000000000000";
    const FAKE_ANTHROPIC: &str = "sk-ant-test00000000000000000000000000";
    const FAKE_GITHUB: &str = "ghp_test0000000000000000000000000000";
    const FAKE_GITLAB: &str = "glpat-test0000000000000000000000000";

    const FAKE_AWS_KEY: &str = "AKIA0000000000000000";
    const FAKE_AWS_SECRET: &str = "aws_secret_access_key=0000000000000000000000000000000000000000";
    const FAKE_JWT: &str = "eyJ0000000000.eyJ0000000000.0000000000000000000000000000";
    const FAKE_STRIPE: &str = "sk_test_0000000000000000000000000000000000000000000000";
    const FAKE_EMAIL: &str = "user@example.com";
    const FAKE_PHONE: &str = "555-123-4567";
    const FAKE_SSN: &str = "123-45-6789";
    const FAKE_CC: &str = "4111111111111111";
    const FAKE_IP4: &str = "192.168.1.100";
    const FAKE_IP6: &str = "2001:0db8:85a3:0000:0000:8a2e:0370:7334";
    const FAKE_DSN: &str = "postgresql://admin:***@db.example.com:5432/mydb";

    // ── secret redaction ─────────────────────────────────────────────────

    #[test]
    fn redact_openai_key() {
        let r = SecretRedactor::secrets_only();
        let input = format!("OPENAI_KEY={}", FAKE_OPENAI);
        let redacted = r.redact(&input);
        assert!(
            !redacted.contains("sk-test"),
            "key should be redacted, got: {redacted}"
        );
        assert!(redacted.contains("[REDACTED:OpenAI API key]"));
    }

    #[test]
    fn redact_anthropic_key() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_ANTHROPIC);
        assert!(!redacted.contains("sk-ant"));
        assert!(redacted.contains("[REDACTED:Anthropic API key]"));
    }

    #[test]
    fn redact_github_token() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_GITHUB);
        assert!(!redacted.contains("ghp_"));
        assert!(redacted.contains("[REDACTED:GitHub token]"));
    }

    #[test]
    fn redact_gitlab_token() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_GITLAB);
        assert!(redacted.contains("[REDACTED:GitLab personal access token]"));
    }

    #[test]
    fn redact_slack_token() {
        // NOTE: Skipped — GitHub push protection blocks even synthetic Slack-format tokens.
        // The Slack regex (xox[bprs]-\d{11,}-\d{11,}-[a-zA-Z0-9]{24,}) is validated by
        // the regex compilation and pattern_count tests.
        // To re-enable, use a string that matches the regex but isn't flagged by GH scanning.
    }

    #[test]
    fn redact_aws_key() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_AWS_KEY);
        assert!(!redacted.contains("AKIA"));
        assert!(redacted.contains("[REDACTED:AWS access key]"));
    }

    #[test]
    fn redact_aws_secret_assignment() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_AWS_SECRET);
        assert!(!redacted.contains("wJalr"));
        assert!(redacted.contains("[REDACTED:AWS secret key]"));
    }

    #[test]
    fn redact_jwt() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_JWT);
        assert!(!redacted.contains("eyJ"));
        assert!(redacted.contains("[REDACTED:JWT token]"));
    }

    #[test]
    fn redact_stripe_key() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_STRIPE);
        assert!(redacted.contains("[REDACTED:Stripe key]"));
    }

    #[test]
    fn redact_db_connection_string() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_DSN);
        assert!(
            !redacted.contains("secretpass"),
            "password should be redacted"
        );
        assert!(redacted.contains("[REDACTED:Database connection string]"));
    }

    #[test]
    fn redact_generic_api_key_assignment() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact("API_KEY=abcdef1234567890abcdef");
        assert!(redacted.contains("[REDACTED:Generic API key assignment]"));
    }

    #[test]
    fn redact_credential_assignment() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact("password=hunter2supersecret");
        assert!(redacted.contains("[REDACTED:Credential assignment]"));
    }

    #[test]
    fn redact_bearer_token_in_header() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact("Authorization: Bearer abcdef1234567890abcdef1234567890abcdef");
        assert!(redacted.contains("[REDACTED:Authorization header]"));
    }

    #[test]
    fn redact_private_key_header() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact("-----BEGIN RSA PRIVATE KEY-----");
        assert!(redacted.contains("[REDACTED:Private key header]"));
    }

    #[test]
    fn redact_pgp_private_key() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact("-----BEGIN PGP PRIVATE KEY BLOCK-----");
        assert!(redacted.contains("[REDACTED:PGP private key]"));
    }

    #[test]
    fn redact_entire_multiline_private_key_blocks() {
        let r = SecretRedactor::secrets_only();
        let pem_body = "synthetic-base64-body-line-1\nsynthetic-base64-body-line-2";
        let pem = format!(
            "before\n-----BEGIN PRIVATE KEY-----\n{pem_body}\n-----END PRIVATE KEY-----\nafter"
        );
        let redacted = r.redact(&pem);
        assert!(!redacted.contains(pem_body));
        assert!(!redacted.contains("END PRIVATE KEY"));
        assert!(redacted.contains("[REDACTED:Private key block]"));

        let pgp_body = "synthetic-pgp-body\nmore-synthetic-pgp-body";
        let pgp = format!(
            "-----BEGIN PGP PRIVATE KEY BLOCK-----\n{pgp_body}\n-----END PGP PRIVATE KEY BLOCK-----"
        );
        let redacted = r.redact(&pgp);
        assert!(!redacted.contains(pgp_body));
        assert!(!redacted.contains("END PGP PRIVATE KEY"));
        assert!(redacted.contains("[REDACTED:PGP private key block]"));
    }

    // ── PII redaction ────────────────────────────────────────────────────

    #[test]
    fn redact_email() {
        let r = SecretRedactor::pii_only();
        let redacted = r.redact(FAKE_EMAIL);
        assert!(!redacted.contains("@"));
        assert!(redacted.contains("[REDACTED:Email address]"));
    }

    #[test]
    fn redact_phone() {
        let r = SecretRedactor::pii_only();
        let redacted = r.redact(FAKE_PHONE);
        assert!(!redacted.contains("555"));
        assert!(redacted.contains("[REDACTED:Phone number]"));
    }

    #[test]
    fn redact_ssn() {
        let r = SecretRedactor::pii_only();
        let redacted = r.redact(FAKE_SSN);
        assert!(!redacted.contains("123-45"));
        assert!(redacted.contains("[REDACTED:SSN]"));
    }

    #[test]
    fn redact_credit_card() {
        let r = SecretRedactor::pii_only();
        let redacted = r.redact(FAKE_CC);
        assert!(!redacted.contains("4111"));
        assert!(redacted.contains("[REDACTED:Credit card number]"));
    }

    #[test]
    fn redact_ipv4() {
        let r = SecretRedactor::pii_only();
        let redacted = r.redact(FAKE_IP4);
        assert!(!redacted.contains("192.168"));
        assert!(redacted.contains("[REDACTED:IPv4 address]"));
    }

    #[test]
    fn redact_ipv6() {
        let r = SecretRedactor::pii_only();
        let redacted = r.redact(FAKE_IP6);
        assert!(!redacted.contains("2001"));
        assert!(redacted.contains("[REDACTED:IPv6 address]"));
    }

    // ── full redaction ───────────────────────────────────────────────────

    #[test]
    fn full_redacts_both_secrets_and_pii() {
        let r = SecretRedactor::full();
        let input = format!("User {} called with key {}", FAKE_EMAIL, FAKE_OPENAI);
        let redacted = r.redact(&input);
        assert!(!redacted.contains("@"));
        assert!(!redacted.contains("sk-test"));
        assert!(redacted.contains("[REDACTED:Email address]"));
        assert!(redacted.contains("[REDACTED:OpenAI API key]"));
    }

    // ── no false positives ───────────────────────────────────────────────

    #[test]
    fn clean_text_unchanged() {
        let r = SecretRedactor::full();
        let input = "The quick brown fox jumps over the lazy dog. File saved to /tmp/output.txt. Status: 200 OK. SHA256: abcdef1234567890.";
        let redacted = r.redact(input);
        assert_eq!(redacted, input);
    }

    #[test]
    fn safe_json_unchanged() {
        let r = SecretRedactor::full();
        let input = r#"{"status":"ok","items":[1,2,3],"message":"done"}"#;
        let redacted = r.redact(input);
        assert_eq!(redacted, input);
    }

    #[test]
    fn harmless_numbers_not_redacted() {
        let r = SecretRedactor::full();
        // 12 digits — shouldn't trigger credit-card because <13 digits
        let input = "Order #123456789012 total: $99.99";
        let redacted = r.redact(input);
        assert_eq!(redacted, input);
    }

    // ── detection ────────────────────────────────────────────────────────

    #[test]
    fn detect_secrets_only() {
        let r = SecretRedactor::secrets_only();
        let secrets = r.detect_secrets(FAKE_OPENAI);
        assert!(!secrets.is_empty());
        assert!(secrets.contains(&"OpenAI API key".to_string()));
    }

    #[test]
    fn detect_pii_only() {
        let r = SecretRedactor::pii_only();
        let pii = r.detect_pii(FAKE_EMAIL);
        assert!(!pii.is_empty());
        assert!(pii.contains(&"Email address".to_string()));
    }

    #[test]
    fn detect_all() {
        let r = SecretRedactor::full();
        let input = format!("{} {}", FAKE_OPENAI, FAKE_EMAIL);
        let all = r.detect(&input);
        assert!(all.len() >= 2);
    }

    #[test]
    fn detect_none_on_clean_text() {
        let r = SecretRedactor::full();
        assert!(r.detect("hello world").is_empty());
        assert!(!r.contains_secrets("hello world"));
    }

    // ── configuration ────────────────────────────────────────────────────

    #[test]
    fn secrets_only_skips_pii() {
        let r = SecretRedactor::secrets_only();
        let redacted = r.redact(FAKE_EMAIL);
        assert_eq!(
            redacted, FAKE_EMAIL,
            "PII should pass through in secrets_only mode"
        );
    }

    #[test]
    fn pii_only_skips_secrets() {
        let r = SecretRedactor::pii_only();
        let redacted = r.redact(FAKE_OPENAI);
        assert_eq!(
            redacted, FAKE_OPENAI,
            "secrets should pass through in pii_only mode"
        );
    }

    #[test]
    fn config_exclude_pattern() {
        let mut config = RedactionConfig::full();
        config.exclude_patterns.push("Email address".to_string());
        let r = SecretRedactor::with_config(config);
        let redacted = r.redact(FAKE_EMAIL);
        assert_eq!(redacted, FAKE_EMAIL, "excluded pattern should not redact");
    }

    #[test]
    fn config_custom_patterns() {
        let mut config = RedactionConfig::default();
        config.custom_patterns.push((
            Regex::new(r"INTERNAL-SECRET-\d+").unwrap(),
            "Internal secret".to_string(),
        ));
        let r = SecretRedactor::with_config(config);
        let redacted = r.redact("Found INTERNAL-SECRET-12345 in logs");
        assert!(redacted.contains("[REDACTED:Internal secret]"));
    }

    // ── JSON redaction ───────────────────────────────────────────────────

    #[test]
    fn redact_json_object() {
        let r = SecretRedactor::full();
        let input = serde_json::json!({
            "api_key": FAKE_OPENAI,
            "user": { "email": FAKE_EMAIL, "name": "bob" },
            "tags": ["safe", FAKE_EMAIL]
        });
        let redacted = r.redact_json(&input);
        let s = redacted.to_string();
        assert!(!s.contains("sk-test"));
        assert!(!s.contains("@"));
        assert!(s.contains("[REDACTED:"));
        assert!(s.contains("bob"), "non-PII values should be preserved");
        assert!(
            s.contains("safe"),
            "non-PII array values should be preserved"
        );
    }

    #[test]
    fn redact_json_nested_array() {
        let r = SecretRedactor::pii_only();
        let input = serde_json::json!([FAKE_EMAIL, "clean", [FAKE_IP4]]);
        let redacted = r.redact_json(&input);
        let s = redacted.to_string();
        assert!(!s.contains("@"));
        assert!(!s.contains("192.168"));
        assert!(s.contains("clean"));
    }

    #[test]
    fn redact_json_uses_sensitive_key_names_for_entire_values() {
        let r = SecretRedactor::secrets_only();
        let password = "correct horse battery staple";
        let authorization = "Basic short-but-sensitive";
        let input = serde_json::json!({
            "password": password,
            "nested": {
                "Authorization": authorization,
                "service_api_key": "punctuation and whitespace !@#$%",
            },
            "array": [{"client-secret": {"deep": "must disappear"}}],
            "safe": "preserved",
        });

        let redacted = r.redact_json(&input);
        let serialized = redacted.to_string();
        assert!(!serialized.contains(password));
        assert!(!serialized.contains(authorization));
        assert!(!serialized.contains("punctuation and whitespace"));
        assert!(!serialized.contains("must disappear"));
        assert!(serialized.contains("[REDACTED:Sensitive JSON field]"));
        assert!(serialized.contains("preserved"));
    }

    #[test]
    fn redact_string_recognizes_quoted_json_keys() {
        let r = SecretRedactor::secrets_only();
        let input = r#"{"password":"long secret with spaces","safe":"visible"}"#;
        let redacted = r.redact(input);
        assert!(!redacted.contains("long secret with spaces"));
        assert!(redacted.contains("[REDACTED:Sensitive JSON field]"));
        assert!(redacted.contains("visible"));
    }

    // ── helper functions ─────────────────────────────────────────────────

    #[test]
    fn redact_and_detect_helper() {
        let r = SecretRedactor::full();
        let input = format!("key={}", FAKE_GITHUB);
        let (safe, detected) = redact_and_detect(&input, &r);
        assert!(!safe.contains("ghp_"));
        assert!(!detected.is_empty());
    }

    #[test]
    fn check_sensitive_helper() {
        let r = SecretRedactor::full();
        assert!(check_sensitive("clean text", &r).is_none());
        assert!(check_sensitive(FAKE_OPENAI, &r).is_some());
    }

    // ── integration-style: multi-line output ─────────────────────────────

    #[test]
    fn redact_multiline_output() {
        let r = SecretRedactor::full();
        let input = format!(
            "STDOUT:\n\
             Connected to {}\n\
             User: {}\n\
             Token: {}\n\
             STDOUT_END",
            FAKE_IP4, FAKE_EMAIL, FAKE_GITHUB,
        );
        let redacted = r.redact(&input);
        assert!(!redacted.contains("192.168"));
        assert!(!redacted.contains("@"));
        assert!(!redacted.contains("ghp_"));
        assert!(redacted.contains("Connected to [REDACTED:IPv4 address]"));
    }

    // ── backward compatibility ───────────────────────────────────────────

    #[test]
    fn legacy_new_is_full() {
        let r = SecretRedactor::new();
        let redacted = r.redact(FAKE_OPENAI);
        assert!(!redacted.contains("sk-test"));
    }

    #[test]
    fn legacy_contains_secrets_works() {
        let r = SecretRedactor::new();
        assert!(r.contains_secrets(FAKE_OPENAI));
        assert!(!r.contains_secrets("clean text"));
    }

    #[test]
    fn legacy_detect_secrets_works() {
        let r = SecretRedactor::new();
        let detected = r.detect_secrets(FAKE_OPENAI);
        assert!(!detected.is_empty());
    }

    #[test]
    fn pattern_count() {
        let full = SecretRedactor::full();
        let secrets = SecretRedactor::secrets_only();
        let pii = SecretRedactor::pii_only();
        assert!(full.pattern_count() > secrets.pattern_count());
        assert!(full.pattern_count() > pii.pattern_count());
        assert!(secrets.pattern_count() > 0);
        assert!(pii.pattern_count() > 0);
    }
}
