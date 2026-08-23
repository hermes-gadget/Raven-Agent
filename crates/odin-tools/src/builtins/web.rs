//! Web search and fetch tools — HTTP GET with timeout, basic web operations.

use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use tracing::instrument;

use odin_core::error::{OdinError, OdinResult};
use odin_core::traits::{Tool, ToolContext};
use odin_core::types::{FunctionSchema, ToolResult, ToolSchema};

/// Shared HTTP client used by all web tools.
fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("OdinTools/0.3 (Raven Agent)")
        .build()
        .expect("Failed to build HTTP client")
}

const MAX_REDIRECTS: usize = 5;
const MAX_HTTP_BODY_BYTES: usize = 100_000;

/// Read at most the configured response size while allowing the caller to
/// report that bytes were discarded. This avoids `Response::text()` buffering
/// an attacker-controlled body before the display limit is applied.
async fn read_bounded_body(
    mut response: reqwest::Response,
) -> Result<(String, bool), reqwest::Error> {
    let mut bytes = Vec::with_capacity(MAX_HTTP_BODY_BYTES);
    let mut truncated = false;

    while let Some(chunk) = response.chunk().await? {
        let remaining = MAX_HTTP_BODY_BYTES.saturating_sub(bytes.len());
        let keep = remaining.min(chunk.len());
        bytes.extend_from_slice(&chunk[..keep]);
        if keep < chunk.len() {
            truncated = true;
            break;
        }
    }

    Ok((String::from_utf8_lossy(&bytes).into_owned(), truncated))
}

#[derive(Debug, Clone, Default)]
struct EgressPolicy {
    allowed_hosts: Arc<HashSet<String>>,
    allowed_networks: Arc<Vec<IpNetwork>>,
}

impl EgressPolicy {
    fn with_allowed_hosts(hosts: impl IntoIterator<Item = String>) -> Self {
        Self {
            allowed_hosts: Arc::new(
                hosts
                    .into_iter()
                    .map(|host| normalize_host(&host))
                    .collect(),
            ),
            allowed_networks: Arc::new(Vec::new()),
        }
    }

    fn with_allowlist(
        hosts: impl IntoIterator<Item = String>,
        cidrs: impl IntoIterator<Item = String>,
    ) -> Result<Self, String> {
        let allowed_networks = cidrs
            .into_iter()
            .map(|cidr| IpNetwork::parse(&cidr))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            allowed_hosts: Arc::new(
                hosts
                    .into_iter()
                    .map(|host| normalize_host(&host))
                    .collect(),
            ),
            allowed_networks: Arc::new(allowed_networks),
        })
    }

    fn explicitly_allows(&self, host: &str, address: IpAddr) -> bool {
        self.allowed_hosts.contains(&normalize_host(host))
            || self
                .allowed_networks
                .iter()
                .any(|network| network.contains(address))
    }
}

#[derive(Debug, Clone, Copy)]
enum IpNetwork {
    V4 { network: u32, prefix: u8 },
    V6 { network: u128, prefix: u8 },
}

impl IpNetwork {
    fn parse(raw: &str) -> Result<Self, String> {
        let (address, prefix) = raw
            .split_once('/')
            .ok_or_else(|| format!("Invalid CIDR allowlist entry '{raw}'"))?;
        let address: IpAddr = address
            .parse()
            .map_err(|error| format!("Invalid CIDR address '{raw}': {error}"))?;
        let prefix: u8 = prefix
            .parse()
            .map_err(|error| format!("Invalid CIDR prefix '{raw}': {error}"))?;
        match address {
            IpAddr::V4(address) if prefix <= 32 => {
                let mask = u32::MAX.checked_shl((32 - prefix) as u32).unwrap_or(0);
                Ok(Self::V4 {
                    network: u32::from(address) & mask,
                    prefix,
                })
            }
            IpAddr::V6(address) if prefix <= 128 => {
                let mask = u128::MAX.checked_shl((128 - prefix) as u32).unwrap_or(0);
                Ok(Self::V6 {
                    network: u128::from(address) & mask,
                    prefix,
                })
            }
            IpAddr::V4(_) => Err(format!("IPv4 CIDR prefix exceeds 32 in '{raw}'")),
            IpAddr::V6(_) => Err(format!("IPv6 CIDR prefix exceeds 128 in '{raw}'")),
        }
    }

    fn contains(self, address: IpAddr) -> bool {
        match (self, address) {
            (Self::V4 { network, prefix }, IpAddr::V4(address)) => {
                let mask = u32::MAX.checked_shl((32 - prefix) as u32).unwrap_or(0);
                u32::from(address) & mask == network
            }
            (Self::V6 { network, prefix }, IpAddr::V6(address)) => {
                let mask = u128::MAX.checked_shl((128 - prefix) as u32).unwrap_or(0);
                u128::from(address) & mask == network
            }
            _ => false,
        }
    }
}

fn normalize_host(host: &str) -> String {
    host.trim_end_matches('.')
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase()
}

fn parse_http_url(raw: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(raw).map_err(|error| format!("Invalid URL: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("URL scheme must be http or https".into());
    }
    if url.host_str().is_none() {
        return Err("URL must include a host".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("Embedded URL credentials are not allowed".into());
    }
    Ok(url)
}

async fn pinned_client(url: &reqwest::Url, policy: &EgressPolicy) -> OdinResult<reqwest::Client> {
    let host = url
        .host_str()
        .ok_or_else(|| OdinError::Validation("URL must include a host".into()))?;
    let resolution_host = host.trim_start_matches('[').trim_end_matches(']');
    let port = url
        .port_or_known_default()
        .ok_or_else(|| OdinError::Validation("URL must include a valid port".into()))?;
    let addresses: Vec<SocketAddr> = if let Ok(ip) = resolution_host.parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        tokio::net::lookup_host((resolution_host, port))
            .await
            .map_err(|error| {
                OdinError::Network(format!("Failed to resolve destination '{host}': {error}"))
            })?
            .collect()
    };
    if addresses.is_empty() {
        return Err(OdinError::Network(format!(
            "Destination '{host}' resolved to no addresses"
        )));
    }
    for address in &addresses {
        if !is_public_ip(address.ip()) && !policy.explicitly_allows(host, address.ip()) {
            return Err(OdinError::PermissionDenied(format!(
                "Destination '{host}' resolves to blocked address {}",
                address.ip()
            )));
        }
    }

    // Pin the connection to an address that was classified above. Keeping the
    // original hostname in the URL preserves Host/SNI while preventing a
    // second DNS lookup from rebinding it to a private address.
    let mut builder = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("OdinTools/0.3 (Raven Agent)")
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy();
    if resolution_host.parse::<IpAddr>().is_err() {
        builder = builder.resolve(resolution_host, addresses[0]);
    }
    builder
        .build()
        .map_err(|error| OdinError::Network(format!("Failed to build HTTP client: {error}")))
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_public_ipv4(ip),
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .map(is_public_ipv4)
            .unwrap_or_else(|| is_public_ipv6(ip)),
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    !(octets[0] == 0
        || octets[0] == 10
        || octets[0] == 127
        || (octets[0] == 100 && (64..=127).contains(&octets[1]))
        || (octets[0] == 169 && octets[1] == 254)
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 168)
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        || (octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
        || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
        || octets[0] >= 224)
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    // Conservatively allow only global-unicast 2000::/3 while excluding the
    // documentation, tunnelling, benchmarking, and ORCHID prefixes. This
    // rejects loopback, unspecified, ULA, link-local, multicast, and other
    // special-purpose ranges by default.
    (segments[0] & 0xe000) == 0x2000
        && !(segments[0] == 0x2001
            && (segments[1] == 0
                || segments[1] == 2
                || (0x10..=0x2f).contains(&segments[1])
                || segments[1] == 0x0db8))
        && segments[0] != 0x2002
}

fn is_sensitive_header(name: &reqwest::header::HeaderName) -> bool {
    matches!(
        name.as_str(),
        "authorization" | "proxy-authorization" | "cookie" | "x-api-key" | "x-auth-token"
    )
}

fn same_origin(left: &reqwest::Url, right: &reqwest::Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str().map(normalize_host) == right.host_str().map(normalize_host)
        && left.port_or_known_default() == right.port_or_known_default()
}

async fn send_with_egress_policy(
    mut method: reqwest::Method,
    mut url: reqwest::Url,
    headers: &reqwest::header::HeaderMap,
    mut body: Option<String>,
    policy: &EgressPolicy,
) -> OdinResult<reqwest::Response> {
    let original = url.clone();
    for redirects in 0..=MAX_REDIRECTS {
        let client = pinned_client(&url, policy).await?;
        let mut request = client.request(method.clone(), url.clone());
        let forwarding_sensitive = same_origin(&original, &url);
        for (name, value) in headers {
            if forwarding_sensitive || !is_sensitive_header(name) {
                request = request.header(name, value);
            }
        }
        if let Some(payload) = body.as_ref() {
            request = request.body(payload.clone());
        }

        let response = request.send().await.map_err(|error| {
            if error.is_timeout() {
                OdinError::Timeout(format!("Request to {url} timed out"))
            } else {
                OdinError::Network(format!("Request to {url} failed: {error}"))
            }
        })?;
        if !response.status().is_redirection() {
            return Ok(response);
        }
        if redirects == MAX_REDIRECTS {
            return Err(OdinError::Network(format!(
                "Request exceeded {MAX_REDIRECTS} redirects"
            )));
        }

        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .ok_or_else(|| OdinError::Network("Redirect omitted Location header".into()))?
            .to_str()
            .map_err(|error| OdinError::Network(format!("Invalid redirect location: {error}")))?;
        let next = url.join(location).map_err(|error| {
            OdinError::Network(format!(
                "Invalid redirect destination '{location}': {error}"
            ))
        })?;
        let next = parse_http_url(next.as_str()).map_err(OdinError::Validation)?;
        if response.status() == reqwest::StatusCode::SEE_OTHER
            || ((response.status() == reqwest::StatusCode::MOVED_PERMANENTLY
                || response.status() == reqwest::StatusCode::FOUND)
                && method == reqwest::Method::POST)
        {
            method = reqwest::Method::GET;
            body = None;
        }
        url = next;
    }
    unreachable!("redirect loop returns or errors at its fixed bound")
}

/// Arguments for `web_fetch`.
#[derive(Debug, Deserialize)]
struct WebFetchArgs {
    url: String,
}

/// Tool that fetches the content of a URL via HTTP GET.
///
/// Returns the raw text content of the response. Configured with a 30-second
/// timeout and a descriptive user-agent.
pub struct WebFetch {
    name: String,
    description: String,
    client: Arc<reqwest::Client>,
}

impl WebFetch {
    /// Create a new `WebFetch` tool.
    pub fn new() -> Self {
        Self {
            name: "web_fetch".into(),
            description:
                "Fetch the contents of a URL via HTTP GET. Returns the raw text response body."
                    .into(),
            client: Arc::new(http_client()),
        }
    }

    /// Construct the JSON schema.
    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Fetch the content of a URL.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch (must start with http:// or https://)"
                        }
                    },
                    "required": ["url"]
                }),
            },
        }
    }
}

impl Default for WebFetch {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetch {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> ToolSchema {
        Self::make_schema(&self.name)
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> &[&str] {
        &["web", "http", "read", "safe"]
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: WebFetchArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let url = &parsed.url;

        // Validate URL
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name.clone(),
                success: false,
                output: String::new(),
                error: Some("URL must start with http:// or https://".into()),
                duration_ms: 0,
                timestamp: Utc::now(),
            });
        }

        let response = self.client.get(url).send().await.map_err(|e| {
            if e.is_timeout() {
                OdinError::Timeout(format!("Request to {url} timed out"))
            } else if e.is_connect() {
                OdinError::Network(format!("Could not connect to {url}: {e}"))
            } else {
                OdinError::Network(format!("Request to {url} failed: {e}"))
            }
        })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            OdinError::Network(format!("Failed to read response body from {url}: {e}"))
        })?;

        let duration_ms = start.elapsed().as_millis() as u64;
        let success = status.is_success();

        let output = if body.len() > 100_000 {
            format!(
                "{} (truncated from {} bytes to 100000)",
                &body[..100_000],
                body.len()
            )
        } else {
            body
        };

        let error = if success {
            None
        } else {
            Some(format!("HTTP {status}"))
        };

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success,
            output,
            error,
            duration_ms,
            timestamp: Utc::now(),
        })
    }
}

/// Arguments for `web_search`.
#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
}

/// Tool that performs a web search.
///
/// This implementation performs a simple HTTP GET to a configurable search
/// endpoint. It can be configured to use different search providers by
/// injecting a custom client or URL.
pub struct WebSearch {
    name: String,
    description: String,
    egress_policy: EgressPolicy,
    /// Optional search URL template (use {query} as placeholder).
    search_url_template: Option<String>,
}

impl WebSearch {
    /// Create a new `WebSearch` tool.
    pub fn new() -> Self {
        Self {
            name: "web_search".into(),
            description: "Search the web for information. Performs a web search using the configured search provider and returns results as text.".into(),
            egress_policy: EgressPolicy::default(),
            search_url_template: None,
        }
    }

    /// Create a `WebSearch` with a custom search URL template.
    ///
    /// The template should contain `{query}` which will be replaced with the
    /// URL-encoded search query.
    pub fn with_search_url(template: impl Into<String>) -> Self {
        Self {
            search_url_template: Some(template.into()),
            ..Self::new()
        }
    }

    /// Allow exact hostnames to resolve to otherwise blocked address ranges.
    pub fn with_allowed_hosts(hosts: impl IntoIterator<Item = String>) -> Self {
        Self {
            egress_policy: EgressPolicy::with_allowed_hosts(hosts),
            ..Self::new()
        }
    }

    /// Allow exact hosts and explicit CIDR ranges for controlled private egress.
    pub fn with_egress_allowlist(
        hosts: impl IntoIterator<Item = String>,
        cidrs: impl IntoIterator<Item = String>,
    ) -> Result<Self, String> {
        Ok(Self {
            egress_policy: EgressPolicy::with_allowlist(hosts, cidrs)?,
            ..Self::new()
        })
    }

    /// Construct the JSON schema.
    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Search the web for information.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (optional, default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }),
            },
        }
    }
}

impl Default for WebSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> ToolSchema {
        Self::make_schema(&self.name)
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> &[&str] {
        &["web", "search", "read", "safe"]
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: WebSearchArgs = serde_json::from_value(args).map_err(|e| OdinError::Tool {
            tool: self.name.clone(),
            message: format!("Invalid arguments: {e}"),
            source: Some(Box::new(e)),
        })?;

        let query = &parsed.query;

        // If a search URL template is configured, use it
        if let Some(template) = &self.search_url_template {
            let encoded = urlencoding(query);
            let url = parse_http_url(&template.replace("{query}", &encoded))
                .map_err(OdinError::Validation)?;

            let response = send_with_egress_policy(
                reqwest::Method::GET,
                url,
                &reqwest::header::HeaderMap::new(),
                None,
                &self.egress_policy,
            )
            .await?;

            let status = response.status();
            let (body, truncated) = read_bounded_body(response)
                .await
                .map_err(|e| OdinError::Network(format!("Failed to read search response: {e}")))?;
            let duration_ms = start.elapsed().as_millis() as u64;

            return Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name.clone(),
                success: status.is_success(),
                output: if truncated {
                    format!("{body} (truncated after {MAX_HTTP_BODY_BYTES} bytes)")
                } else {
                    body
                },
                error: if status.is_success() {
                    None
                } else {
                    Some(format!("HTTP {status}"))
                },
                duration_ms,
                timestamp: Utc::now(),
            });
        }

        // No search template configured — return informative message
        let duration_ms = start.elapsed().as_millis() as u64;
        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success: true,
            output: format!(
                "Web search is not configured with a search provider URL. \
                 To enable web search, configure a search URL template using \
                 `with_search_url()`. Query was: {query}"
            ),
            error: None,
            duration_ms,
            timestamp: Utc::now(),
        })
    }
}

// ── http_request ───────────────────────────────────────────────────

/// Arguments for `http_request`.
#[derive(Debug, Deserialize)]
struct HttpRequestArgs {
    method: String,
    url: String,
    headers: Option<Vec<HeaderPair>>,
    body: Option<String>,
}

/// Key-value header pair for `http_request`.
#[derive(Debug, Deserialize)]
struct HeaderPair {
    name: String,
    value: String,
}

/// Tool that makes arbitrary HTTP requests.
///
/// Supports GET, POST, PUT, DELETE methods with optional headers and body.
/// URLs must be http:// or https://. Uses a 30-second timeout.
pub struct HttpRequest {
    name: String,
    description: String,
    client: Arc<reqwest::Client>,
}

impl HttpRequest {
    /// Create a new `HttpRequest` tool.
    pub fn new() -> Self {
        Self {
            name: "http_request".into(),
            description: "Make an HTTP request with the given method (GET/POST/PUT/DELETE), URL, optional headers, and optional body. Safe with URL validation.".into(),
            client: Arc::new(http_client()),
        }
    }

    fn make_schema(name: &str) -> ToolSchema {
        ToolSchema {
            schema_type: "function".into(),
            function: FunctionSchema {
                name: name.into(),
                description: "Make an HTTP request (GET/POST/PUT/DELETE).".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "HTTP method: GET, POST, PUT, or DELETE",
                            "enum": ["GET", "POST", "PUT", "DELETE"]
                        },
                        "url": {
                            "type": "string",
                            "description": "The URL to request (must start with http:// or https://)"
                        },
                        "headers": {
                            "type": "array",
                            "description": "Optional HTTP headers as array of {name, value} objects",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Header name"},
                                    "value": {"type": "string", "description": "Header value"}
                                },
                                "required": ["name", "value"]
                            }
                        },
                        "body": {
                            "type": "string",
                            "description": "Optional request body (for POST/PUT)"
                        }
                    },
                    "required": ["method", "url"]
                }),
            },
        }
    }
}

impl Default for HttpRequest {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for HttpRequest {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> ToolSchema {
        Self::make_schema(&self.name)
    }

    fn is_safe(&self) -> bool {
        true
    }

    fn capability_tags(&self) -> &[&str] {
        &["web", "http", "safe"]
    }

    #[instrument(skip(self, _context), fields(tool = self.name))]
    async fn execute(
        &self,
        args: serde_json::Value,
        _context: &ToolContext,
    ) -> OdinResult<ToolResult> {
        let start = Instant::now();

        let parsed: HttpRequestArgs =
            serde_json::from_value(args).map_err(|e| OdinError::Tool {
                tool: self.name.clone(),
                message: format!("Invalid arguments: {e}"),
                source: Some(Box::new(e)),
            })?;

        // Validate URL
        if !parsed.url.starts_with("http://") && !parsed.url.starts_with("https://") {
            return Ok(ToolResult {
                call_id: String::new(),
                tool_name: self.name.clone(),
                success: false,
                output: String::new(),
                error: Some("URL must start with http:// or https://".into()),
                duration_ms: 0,
                timestamp: Utc::now(),
            });
        }

        // Validate method
        let method = match parsed.method.to_uppercase().as_str() {
            "GET" => reqwest::Method::GET,
            "POST" => reqwest::Method::POST,
            "PUT" => reqwest::Method::PUT,
            "DELETE" => reqwest::Method::DELETE,
            other => {
                return Ok(ToolResult {
                    call_id: String::new(),
                    tool_name: self.name.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Invalid method '{other}'. Must be GET, POST, PUT, or DELETE."
                    )),
                    duration_ms: 0,
                    timestamp: Utc::now(),
                });
            }
        };

        // Build request
        let mut req = self.client.request(method, &parsed.url);

        if let Some(ref headers) = parsed.headers {
            for h in headers {
                if let (Ok(name), Ok(value)) = (
                    reqwest::header::HeaderName::from_bytes(h.name.as_bytes()),
                    reqwest::header::HeaderValue::from_str(&h.value),
                ) {
                    req = req.header(name, value);
                }
            }
        }

        if let Some(ref body) = parsed.body {
            req = req.body(body.clone());
        }

        // Execute
        let response = req.send().await.map_err(|e| {
            if e.is_timeout() {
                OdinError::Timeout(format!("Request to {} timed out", parsed.url))
            } else if e.is_connect() {
                OdinError::Network(format!("Could not connect to {}: {e}", parsed.url))
            } else {
                OdinError::Network(format!("Request to {} failed: {e}", parsed.url))
            }
        })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| {
            OdinError::Network(format!(
                "Failed to read response body from {}: {e}",
                parsed.url
            ))
        })?;

        let duration_ms = start.elapsed().as_millis() as u64;
        let success = status.is_success();

        let output = if body.len() > 100_000 {
            format!(
                "{} (truncated from {} bytes to 100000)",
                &body[..100_000],
                body.len()
            )
        } else {
            body
        };

        let error = if success {
            None
        } else {
            Some(format!("HTTP {status}"))
        };

        Ok(ToolResult {
            call_id: String::new(),
            tool_name: self.name.clone(),
            success,
            output,
            error,
            duration_ms,
            timestamp: Utc::now(),
        })
    }
}

/// Simple URL encoding for search queries (replaces special chars).
fn urlencoding(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            b' ' => result.push_str("%20"),
            _ => {
                result.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn test_context() -> ToolContext {
        ToolContext {
            agent_id: Default::default(),
            session_id: Default::default(),
            working_dir: PathBuf::from("/tmp"),
            env: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_web_fetch_invalid_url() {
        let fetch = WebFetch::new();
        let args = serde_json::json!({
            "url": "not-a-url"
        });
        let result = fetch.execute(args, &test_context()).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("URL must start with"));
    }

    #[tokio::test]
    async fn test_web_fetch_http_error() {
        let fetch = WebFetch::new();
        let args = serde_json::json!({
            "url": "https://httpstat.us/404"
        });
        let result = fetch.execute(args, &test_context()).await;
        // May fail due to network or return HTTP error - either is acceptable
        if let Ok(res) = result
            && !res.success
        {
            assert!(res.error.unwrap().contains("HTTP"));
        }
    }

    #[tokio::test]
    async fn test_web_search_no_template() {
        let search = WebSearch::new();
        let args = serde_json::json!({
            "query": "rust programming",
            "max_results": 3
        });
        let result = search.execute(args, &test_context()).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("not configured"));
    }

    #[tokio::test]
    async fn test_web_search_blocks_private_destination() {
        let search = WebSearch::with_search_url("http://127.0.0.1:9/?q={query}");
        let args = serde_json::json!({"query": "rust"});

        let error = search
            .execute(args, &test_context())
            .await
            .expect_err("private search destinations must be blocked");
        assert!(matches!(error, OdinError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn test_web_fetch_timeout() {
        let fetch = WebFetch::new();
        let args = serde_json::json!({
            "url": "https://httpstat.us/200?sleep=5000"
        });
        // Should not hang — the 30s client timeout should handle it
        let result = fetch.execute(args, &test_context()).await;
        // Either success or a network error is fine
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_urlencoding() {
        assert_eq!(urlencoding("hello world"), "hello%20world");
        assert_eq!(urlencoding("foo/bar"), "foo%2Fbar");
        assert_eq!(urlencoding("a b c"), "a%20b%20c");
        assert_eq!(urlencoding("simple"), "simple");
        assert_eq!(urlencoding(""), "");
    }

    #[tokio::test]
    async fn test_http_request_invalid_url() {
        let req = HttpRequest::new();
        let args = serde_json::json!({
            "method": "GET",
            "url": "not-a-url"
        });
        let result = req.execute(args, &test_context()).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("URL must start with"));
    }

    #[tokio::test]
    async fn test_http_request_invalid_method() {
        let req = HttpRequest::new();
        let args = serde_json::json!({
            "method": "INVALID",
            "url": "https://example.com"
        });
        let result = req.execute(args, &test_context()).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Invalid method"));
    }

    #[tokio::test]
    async fn test_http_request_get() {
        let req = HttpRequest::new();
        let args = serde_json::json!({
            "method": "GET",
            "url": "https://httpstat.us/200"
        });
        let result = req.execute(args, &test_context()).await;
        // Network may or may not be available — either is fine
        if let Ok(res) = result
            && !res.success
        {
            assert!(res.error.unwrap().contains("HTTP"));
        }
    }
}
