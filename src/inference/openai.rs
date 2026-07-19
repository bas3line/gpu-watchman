//! Privacy-safe `OpenAI` chat-completions requests and bounded response parsing.

use std::io::{BufRead, BufReader, Read};
use std::net::IpAddr;
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use reqwest::blocking::{Client, Response};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::redirect::Policy;
use serde_json::{Value, json};
use url::Url;

use crate::domain::{CanaryAttempt, CanaryFailure, CanaryFailureStage, CanaryTarget};

const MAX_FAILURE_BYTES: usize = 512;
const MAX_BODY_BYTES: usize = 8 << 20;
pub(crate) const MAX_REPORTED_PROMPT_TOKENS_PER_REQUEST: u64 = 10_000_000;

// Conservative admission-control constants for the allocations made by one
// active request. A byte of dense JSON can expand into a `Value`, collection
// capacity, and allocator bookkeeping that are many times larger than its wire
// representation. The 64x allowance also covers transient collection growth;
// raw body/line/event buffers are counted separately below.
pub(crate) const JSON_VALUE_MEMORY_MULTIPLIER: u64 = 64;
pub(crate) const WORKER_FIXED_MEMORY_BYTES: u64 = 1 << 20;
const JSON_ESCAPE_MEMORY_MULTIPLIER: u64 = 6;
const REQUEST_BUFFER_CAPACITY_MULTIPLIER: u64 = 2;
const REQUEST_ENVELOPE_MEMORY_BYTES: u64 = 512;
const JSON_RESPONSE_BUFFER_CAPACITY_COPIES: u64 = 2;
const STREAM_RESPONSE_BUFFER_CAPACITY_COPIES: u64 = 4;

/// Immutable request configuration. `Debug` is intentionally omitted because
/// this type owns prompt content and an optional credential.
#[derive(Clone)]
pub(crate) struct OpenAiClientOptions {
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) prompt: String,
    pub(crate) expectation: Option<String>,
    pub(crate) max_tokens: u32,
    pub(crate) timeout: Duration,
    pub(crate) max_body_bytes: usize,
    pub(crate) stream: bool,
    pub(crate) allow_insecure_http: bool,
}

/// Reusable blocking client for a bounded canary run.
pub(crate) struct OpenAiClient {
    client: Client,
    url: Url,
    safe_url: String,
    model: String,
    expectation: Option<String>,
    request_body: Vec<u8>,
    max_tokens: u32,
    max_body_bytes: usize,
    stream: bool,
}

/// Bound the encoded request without retaining another serialized copy.
/// `serde_json` emits at most six bytes for each input byte (`\u00xx`), the
/// fixed envelope allowance covers field names, punctuation, booleans, and the
/// decimal token count, and a second copy covers `Vec` capacity growth.
pub(crate) fn request_body_memory_budget(model_bytes: usize, prompt_bytes: usize) -> Option<u64> {
    let input_bytes = u64::try_from(model_bytes)
        .ok()?
        .checked_add(u64::try_from(prompt_bytes).ok()?)?;
    let encoded_bytes = input_bytes
        .checked_mul(JSON_ESCAPE_MEMORY_MULTIPLIER)?
        .checked_add(REQUEST_ENVELOPE_MEMORY_BYTES)?;
    encoded_bytes.checked_mul(REQUEST_BUFFER_CAPACITY_MULTIPLIER)
}

/// Estimate the peak allocations attributable to one worker. The request body
/// is cloned by `reqwest`; response parsing retains one raw JSON body, or a
/// stream line plus assembled event, while a `serde_json::Value` is live. Each
/// matcher owns the expectation bytes plus one `usize` prefix-table entry per
/// byte.
pub(crate) fn worker_memory_budget(
    request_body_bytes: u64,
    response_limit_bytes: usize,
    expectation_bytes: usize,
    stream: bool,
) -> Option<u64> {
    let response_limit_bytes = u64::try_from(response_limit_bytes).ok()?;
    let response_buffer_capacity_copies = if stream {
        STREAM_RESPONSE_BUFFER_CAPACITY_COPIES
    } else {
        JSON_RESPONSE_BUFFER_CAPACITY_COPIES
    };
    let response_wire_bytes = response_limit_bytes.checked_add(1)?;
    let response_bytes = response_wire_bytes
        .checked_mul(JSON_VALUE_MEMORY_MULTIPLIER.checked_add(response_buffer_capacity_copies)?)?;
    let prefix_entry_bytes = u64::try_from(std::mem::size_of::<usize>()).ok()?;
    let matcher_bytes = u64::try_from(expectation_bytes)
        .ok()?
        .checked_mul(prefix_entry_bytes.checked_add(1)?)?;
    request_body_bytes
        .checked_add(response_bytes)?
        .checked_add(matcher_bytes)?
        .checked_add(WORKER_FIXED_MEMORY_BYTES)
}

impl OpenAiClient {
    pub(crate) fn new(options: OpenAiClientOptions) -> Result<Self> {
        let url = resolve_chat_completions_url(&options.base_url)?;
        validate_transport_security(&url, options.allow_insecure_http)?;
        if options.max_body_bytes == 0 || options.max_body_bytes > MAX_BODY_BYTES {
            bail!("canary response limit must be between 1 byte and 8 MiB");
        }
        let safe_url = safe_origin(&url);
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("gpu-watchman"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            ACCEPT,
            HeaderValue::from_static(if options.stream {
                "text/event-stream"
            } else {
                "application/json"
            }),
        );
        if let Some(api_key) = options.api_key.as_deref() {
            let value = HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("API key contains invalid HTTP header characters")?;
            headers.insert(AUTHORIZATION, value);
        }
        let mut client = Client::builder()
            .timeout(options.timeout)
            .connect_timeout(options.timeout.min(Duration::from_secs(10)))
            .redirect(Policy::none())
            .default_headers(headers);
        // A syntactically local target must never be routed through ambient
        // HTTP_PROXY/ALL_PROXY configuration with its prompt and bearer token.
        if is_loopback_target(&url) {
            client = client.no_proxy();
        }
        let client = client
            .build()
            .context("build OpenAI-compatible HTTP client")?;
        let mut request = json!({
            "model": &options.model,
            "messages": [{"role": "user", "content": &options.prompt}],
            "max_tokens": options.max_tokens,
            "temperature": 0,
            "stream": options.stream,
        });
        if options.stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        let request_body =
            serde_json::to_vec(&request).context("encode OpenAI-compatible request")?;
        let request_body_budget =
            request_body_memory_budget(options.model.len(), options.prompt.len())
                .context("canary request memory budget overflow")?;
        if u64::try_from(request_body.len()).unwrap_or(u64::MAX) > request_body_budget {
            bail!("encoded canary request exceeded its internal memory budget");
        }

        Ok(Self {
            client,
            url,
            safe_url,
            model: options.model,
            expectation: options.expectation,
            request_body,
            max_tokens: options.max_tokens,
            max_body_bytes: options.max_body_bytes,
            stream: options.stream,
        })
    }

    pub(crate) fn target(&self) -> CanaryTarget {
        CanaryTarget {
            url: self.safe_url.clone(),
            route: "chat_completions".to_owned(),
            model: self.model.clone(),
            stream: self.stream,
        }
    }

    pub(crate) fn execute(&self, index: u32) -> CanaryAttempt {
        let started = Instant::now();
        let response = match self
            .client
            .post(self.url.clone())
            .body(self.request_body.clone())
            .send()
        {
            Ok(response) => response,
            Err(error) => {
                let message = if error.is_timeout() {
                    "request timed out"
                } else if error.is_connect() {
                    "connection failed"
                } else {
                    "request transport failed"
                };
                return failed_attempt(
                    index,
                    CanaryFailureStage::Transport,
                    message,
                    0,
                    None,
                    Some(elapsed_milliseconds(started)),
                );
            }
        };
        let headers_ms = elapsed_milliseconds(started);
        let status_code = response.status().as_u16();
        if !response.status().is_success() {
            return failed_attempt(
                index,
                CanaryFailureStage::Http,
                format!("HTTP status {status_code}"),
                status_code,
                Some(headers_ms),
                Some(elapsed_milliseconds(started)),
            );
        }

        let parsed = if self.stream {
            parse_stream(
                response,
                started,
                self.max_body_bytes,
                self.expectation.as_deref(),
            )
        } else {
            parse_json(response, self.max_body_bytes, self.expectation.as_deref())
        };
        match parsed {
            Ok(parsed) => Self::finish_attempt(
                index,
                status_code,
                headers_ms,
                started,
                parsed,
                self.max_tokens,
            ),
            Err(failure) => failed_attempt(
                index,
                failure.stage,
                failure.message,
                status_code,
                Some(headers_ms),
                Some(elapsed_milliseconds(started)),
            ),
        }
    }

    fn finish_attempt(
        index: u32,
        status_code: u16,
        headers_ms: f64,
        started: Instant,
        parsed: ParsedCompletion,
        max_tokens: u32,
    ) -> CanaryAttempt {
        let e2e = started.elapsed();
        let e2e_ms = duration_milliseconds(e2e);
        if !parsed.output.has_non_whitespace {
            return failed_attempt(
                index,
                CanaryFailureStage::EmptyOutput,
                "response contained no generated content",
                status_code,
                Some(headers_ms),
                Some(e2e_ms),
            );
        }
        let expectation_met = parsed.output.expectation_met;
        let success = expectation_met.unwrap_or(true);
        let prompt_tokens = parsed
            .prompt_tokens
            .filter(|tokens| (1..=MAX_REPORTED_PROMPT_TOKENS_PER_REQUEST).contains(tokens));
        let completion_tokens = parsed
            .completion_tokens
            .filter(|tokens| (1..=u64::from(max_tokens)).contains(tokens));
        let output_tokens_per_second = output_token_rate(completion_tokens, parsed.ttft, e2e);
        CanaryAttempt {
            index,
            success,
            status_code,
            headers_ms: Some(headers_ms),
            ttft_ms: parsed.ttft.map(duration_milliseconds),
            e2e_ms: Some(e2e_ms),
            prompt_tokens,
            completion_tokens,
            output_tokens_per_second,
            // The requested model is recorded once in report.target. Repeating
            // it per attempt would multiply user-controlled identity data by
            // the request count without adding evidence.
            model: String::new(),
            finish_reason: parsed.finish_reason.unwrap_or_default(),
            expectation_met,
            failure: (!success).then(|| CanaryFailure {
                stage: CanaryFailureStage::Expectation,
                message: "generated content did not satisfy the configured expectation".to_owned(),
            }),
        }
    }
}

#[derive(Debug)]
struct ParsedCompletion {
    output: OutputMatcher,
    finish_reason: Option<String>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    ttft: Option<Duration>,
}

impl ParsedCompletion {
    fn new(expectation: Option<&str>) -> Self {
        Self {
            output: OutputMatcher::new(expectation),
            finish_reason: None,
            prompt_tokens: None,
            completion_tokens: None,
            ttft: None,
        }
    }
}

/// Incremental substring matcher. It retains only the configured expectation
/// and its prefix table, never the model's generated text.
#[derive(Debug)]
struct OutputMatcher {
    needle: Option<Vec<u8>>,
    prefix: Vec<usize>,
    matched: usize,
    has_non_whitespace: bool,
    expectation_met: Option<bool>,
}

impl OutputMatcher {
    fn new(expectation: Option<&str>) -> Self {
        let needle = expectation.map(str::as_bytes).map(ToOwned::to_owned);
        let prefix = needle.as_deref().map_or_else(Vec::new, prefix_table);
        Self {
            expectation_met: needle.as_ref().map(Vec::is_empty),
            needle,
            prefix,
            matched: 0,
            has_non_whitespace: false,
        }
    }

    fn push(&mut self, piece: &str) {
        self.has_non_whitespace |= piece.chars().any(|character| !character.is_whitespace());
        if self.expectation_met != Some(false) {
            return;
        }
        let Some(needle) = self.needle.as_deref() else {
            return;
        };
        for &byte in piece.as_bytes() {
            while self.matched > 0 && needle[self.matched] != byte {
                self.matched = self.prefix[self.matched - 1];
            }
            if needle[self.matched] == byte {
                self.matched += 1;
            }
            if self.matched == needle.len() {
                self.expectation_met = Some(true);
                return;
            }
        }
    }
}

fn prefix_table(needle: &[u8]) -> Vec<usize> {
    let mut table = vec![0; needle.len()];
    let mut matched = 0;
    for index in 1..needle.len() {
        while matched > 0 && needle[matched] != needle[index] {
            matched = table[matched - 1];
        }
        if needle[matched] == needle[index] {
            matched += 1;
            table[index] = matched;
        }
    }
    table
}

fn parse_stream(
    response: Response,
    started: Instant,
    max_body_bytes: usize,
    expectation: Option<&str>,
) -> Result<ParsedCompletion, CanaryFailure> {
    reject_oversized_content_length(&response, max_body_bytes)?;
    let limit = u64::try_from(max_body_bytes.saturating_add(1)).unwrap_or(u64::MAX);
    parse_stream_reader(response.take(limit), started, max_body_bytes, expectation)
}

fn parse_stream_reader(
    reader: impl Read,
    started: Instant,
    max_body_bytes: usize,
    expectation: Option<&str>,
) -> Result<ParsedCompletion, CanaryFailure> {
    let mut reader = BufReader::new(reader);
    let mut parsed = ParsedCompletion::new(expectation);
    let mut line = String::new();
    let mut event = String::new();
    let mut bytes = 0_usize;
    loop {
        line.clear();
        let read = reader.read_line(&mut line).map_err(|error| {
            if error.kind() == std::io::ErrorKind::InvalidData {
                protocol_failure("stream response was not valid UTF-8")
            } else if matches!(
                error.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) {
                transport_failure("stream response timed out")
            } else {
                transport_failure("could not read stream response")
            }
        })?;
        if read == 0 {
            if !event.is_empty() && process_stream_event(&event, started, &mut parsed)? {
                return Ok(parsed);
            }
            break;
        }
        bytes = bytes.saturating_add(read);
        if bytes > max_body_bytes {
            return Err(protocol_failure(
                "response exceeds the configured size limit",
            ));
        }
        let value = line.trim_end_matches(['\r', '\n']);
        if value.is_empty() {
            if !event.is_empty() {
                let done = process_stream_event(&event, started, &mut parsed)?;
                event.clear();
                if done {
                    return Ok(parsed);
                }
            }
            continue;
        }
        if value.starts_with(':') {
            continue;
        }
        if let Some(data) = value.strip_prefix("data:") {
            if !event.is_empty() {
                event.push('\n');
            }
            event.push_str(data.trim_start());
        }
    }
    if parsed.finish_reason.is_none() {
        return Err(protocol_failure(
            "stream ended before a terminal completion event",
        ));
    }
    Ok(parsed)
}

fn process_stream_event(
    event: &str,
    started: Instant,
    parsed: &mut ParsedCompletion,
) -> Result<bool, CanaryFailure> {
    if event.trim() == "[DONE]" {
        return Ok(true);
    }
    let value: Value = serde_json::from_str(event)
        .map_err(|_| protocol_failure("stream contained malformed JSON"))?;
    if value.get("error").is_some() {
        return Err(protocol_failure("stream returned an error object"));
    }
    update_metadata(&value, parsed);
    if let Some(choice) = single_choice(&value, true)? {
        if parsed.finish_reason.is_none() {
            parsed.finish_reason = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .and_then(safe_finish_reason);
        }
        if let Some(delta) = choice.get("delta") {
            for field in ["reasoning", "reasoning_content"] {
                if let Some(reasoning) = delta.get(field) {
                    visit_content_text(reasoning, &mut |piece| {
                        if !piece.is_empty() {
                            // Reasoning tokens are part of authoritative completion
                            // usage and generation time, but must never satisfy the
                            // configured final-content expectation.
                            parsed.ttft.get_or_insert_with(|| started.elapsed());
                        }
                    });
                }
            }
        }
        if let Some(content) = choice.get("delta").and_then(|delta| delta.get("content")) {
            visit_content_text(content, &mut |piece| {
                if !piece.is_empty() {
                    parsed.ttft.get_or_insert_with(|| started.elapsed());
                    parsed.output.push(piece);
                }
            });
        }
    }
    Ok(false)
}

fn parse_json(
    response: Response,
    max_body_bytes: usize,
    expectation: Option<&str>,
) -> Result<ParsedCompletion, CanaryFailure> {
    reject_oversized_content_length(&response, max_body_bytes)?;
    let limit = u64::try_from(max_body_bytes.saturating_add(1)).unwrap_or(u64::MAX);
    parse_json_reader(response.take(limit), max_body_bytes, expectation)
}

fn parse_json_reader(
    mut reader: impl Read,
    max_body_bytes: usize,
    expectation: Option<&str>,
) -> Result<ParsedCompletion, CanaryFailure> {
    let mut body = Vec::new();
    reader
        .read_to_end(&mut body)
        .map_err(|_| transport_failure("could not read JSON response"))?;
    if body.len() > max_body_bytes {
        return Err(protocol_failure(
            "response exceeds the configured size limit",
        ));
    }
    let value: Value = serde_json::from_slice(&body)
        .map_err(|_| protocol_failure("response contained malformed JSON"))?;
    drop(body);
    if value.get("error").is_some() {
        return Err(protocol_failure("response returned an error object"));
    }
    let mut parsed = ParsedCompletion::new(expectation);
    update_metadata(&value, &mut parsed);
    if let Some(choice) = single_choice(&value, false)? {
        if parsed.finish_reason.is_none() {
            parsed.finish_reason = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .and_then(safe_finish_reason);
        }
        if let Some(content) = choice
            .get("message")
            .and_then(|message| message.get("content"))
            .or_else(|| choice.get("text"))
        {
            visit_content_text(content, &mut |piece| parsed.output.push(piece));
        }
    }
    Ok(parsed)
}

fn update_metadata(value: &Value, parsed: &mut ParsedCompletion) {
    if let Some(usage) = value.get("usage") {
        if let Some(prompt_tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) {
            parsed.prompt_tokens = Some(prompt_tokens);
        }
        if let Some(completion_tokens) = usage.get("completion_tokens").and_then(Value::as_u64) {
            parsed.completion_tokens = Some(completion_tokens);
        }
    }
}

fn reject_oversized_content_length(
    response: &Response,
    max_body_bytes: usize,
) -> Result<(), CanaryFailure> {
    let maximum = u64::try_from(max_body_bytes).unwrap_or(u64::MAX);
    if response
        .content_length()
        .is_some_and(|length| length > maximum)
    {
        return Err(protocol_failure(
            "response exceeds the configured size limit",
        ));
    }
    Ok(())
}

fn single_choice(value: &Value, allow_empty: bool) -> Result<Option<&Value>, CanaryFailure> {
    let Some(choices) = value.get("choices") else {
        return Ok(None);
    };
    let Some(choices) = choices.as_array() else {
        return Err(protocol_failure("response choices must be an array"));
    };
    if choices.is_empty() && allow_empty {
        return Ok(None);
    }
    if choices.len() != 1 {
        return Err(protocol_failure(
            "canary response must contain exactly one completion choice",
        ));
    }
    let choice = &choices[0];
    if !choice.is_object() {
        return Err(protocol_failure("canary response choice must be an object"));
    }
    if choice
        .get("index")
        .is_some_and(|index| index.as_u64() != Some(0))
    {
        return Err(protocol_failure(
            "canary response choice must use index zero",
        ));
    }
    Ok(Some(choice))
}

fn safe_finish_reason(value: &str) -> Option<String> {
    matches!(
        value,
        "stop" | "length" | "tool_calls" | "content_filter" | "function_call"
    )
    .then(|| value.to_owned())
}

fn visit_content_text(value: &Value, visit: &mut impl FnMut(&str)) {
    match value {
        Value::String(value) => visit(value),
        Value::Array(values) => {
            for value in values {
                visit_content_text(value, visit);
            }
        }
        Value::Object(object) => {
            if let Some(value) = object.get("text") {
                visit_content_text(value, visit);
            }
        }
        _ => {}
    }
}

fn output_token_rate(tokens: Option<u64>, ttft: Option<Duration>, e2e: Duration) -> Option<f64> {
    let tokens = tokens?;
    let ttft = ttft?;
    let generated_tokens = tokens.saturating_sub(1);
    let generation = e2e.saturating_sub(ttft);
    if generated_tokens == 0 || generation.is_zero() {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    Some(generated_tokens as f64 / generation.as_secs_f64())
}

fn failed_attempt(
    index: u32,
    stage: CanaryFailureStage,
    message: impl AsRef<str>,
    status_code: u16,
    headers_ms: Option<f64>,
    e2e_ms: Option<f64>,
) -> CanaryAttempt {
    CanaryAttempt {
        index,
        success: false,
        status_code,
        headers_ms,
        ttft_ms: None,
        e2e_ms,
        prompt_tokens: None,
        completion_tokens: None,
        output_tokens_per_second: None,
        model: String::new(),
        finish_reason: String::new(),
        expectation_met: None,
        failure: Some(CanaryFailure {
            stage,
            message: bounded_failure(message.as_ref()),
        }),
    }
}

fn protocol_failure(message: &str) -> CanaryFailure {
    CanaryFailure {
        stage: CanaryFailureStage::Protocol,
        message: bounded_failure(message),
    }
}

fn transport_failure(message: &str) -> CanaryFailure {
    CanaryFailure {
        stage: CanaryFailureStage::Transport,
        message: bounded_failure(message),
    }
}

fn bounded_failure(message: &str) -> String {
    let message = message.replace(['\r', '\n'], " ");
    if message.len() <= MAX_FAILURE_BYTES {
        return message;
    }
    let mut end = MAX_FAILURE_BYTES.saturating_sub(3);
    while !message.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{}...", &message[..end])
}

fn resolve_chat_completions_url(base: &str) -> Result<Url> {
    let mut url = Url::parse(base.trim()).context("invalid canary base URL")?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("canary base URL must use http or https");
    }
    if url.host_str().is_none() {
        bail!("canary base URL must include a host");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("canary base URL must not include credentials");
    }
    url.set_fragment(None);
    let path = url.path().trim_end_matches('/').to_owned();
    if path.ends_with("/chat/completions") {
        url.set_path(&path);
    } else {
        let path = if path.is_empty() {
            "/v1/chat/completions".to_owned()
        } else if path.ends_with("/v1") {
            format!("{path}/chat/completions")
        } else {
            format!("{path}/v1/chat/completions")
        };
        url.set_path(&path);
    }
    Ok(url)
}

fn validate_transport_security(url: &Url, allow_insecure_http: bool) -> Result<()> {
    if url.scheme() != "http" || allow_insecure_http {
        return Ok(());
    }
    if !is_loopback_target(url) {
        bail!("remote canary HTTP requires --allow-insecure-http; use HTTPS when possible");
    }
    Ok(())
}

fn is_loopback_target(url: &Url) -> bool {
    let host = url.host_str().unwrap_or_default();
    let local_name = matches!(
        host.to_ascii_lowercase().as_str(),
        "localhost" | "localhost."
    );
    let loopback_ip = IpAddr::from_str(host).is_ok_and(|address| address.is_loopback());
    local_name || loopback_ip
}

fn safe_origin(url: &Url) -> String {
    url.origin().ascii_serialization()
}

fn elapsed_milliseconds(started: Instant) -> f64 {
    duration_milliseconds(started.elapsed())
}

fn duration_milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_base_and_full_chat_urls() {
        assert_eq!(
            resolve_chat_completions_url("http://localhost:8000/v1")
                .unwrap()
                .as_str(),
            "http://localhost:8000/v1/chat/completions"
        );
        assert_eq!(
            resolve_chat_completions_url("https://host/prefix/v1/chat/completions/")
                .unwrap()
                .as_str(),
            "https://host/prefix/v1/chat/completions"
        );
        assert!(resolve_chat_completions_url("file:///tmp/model").is_err());
        assert!(resolve_chat_completions_url("http://:secret@host/v1").is_err());
    }

    #[test]
    fn reports_only_the_safe_target_origin() {
        let url = Url::parse("https://host/prefix/private/v1?secret_key=private").unwrap();
        assert_eq!(safe_origin(&url), "https://host");
    }

    #[test]
    fn cleartext_http_is_local_only_without_explicit_opt_in() {
        let local = Url::parse("http://127.0.0.2:8000/v1").unwrap();
        let remote = Url::parse("http://inference.example/v1").unwrap();
        let secure = Url::parse("https://inference.example/v1").unwrap();

        assert!(validate_transport_security(&local, false).is_ok());
        assert!(is_loopback_target(&local));
        assert!(validate_transport_security(&secure, false).is_ok());
        assert!(validate_transport_security(&remote, false).is_err());
        assert!(validate_transport_security(&remote, true).is_ok());
    }

    #[test]
    fn token_rate_requires_authoritative_usage_and_stream_timing() {
        assert_eq!(
            output_token_rate(
                Some(5),
                Some(Duration::from_millis(100)),
                Duration::from_millis(500)
            ),
            Some(10.0)
        );
        assert_eq!(
            output_token_rate(
                None,
                Some(Duration::from_millis(100)),
                Duration::from_millis(500)
            ),
            None
        );
        assert_eq!(
            output_token_rate(Some(5), None, Duration::from_millis(500)),
            None
        );
    }

    #[test]
    fn parses_role_content_and_usage_sse_events_with_crlf_and_comments() {
        let body = concat!(
            ": keepalive\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},",
            "\"finish_reason\":null}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"gpu-\"},",
            "\"finish_reason\":null}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"watchman-ok\"},",
            "\"finish_reason\":\"stop\"}]}\r\n\r\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,",
            "\"completion_tokens\":3}}\r\n\r\n",
            "data: [DONE]\r\n\r\n"
        );

        let parsed = parse_stream_reader(
            std::io::Cursor::new(body),
            Instant::now(),
            body.len(),
            Some("gpu-watchman-ok"),
        )
        .unwrap();

        assert!(parsed.output.has_non_whitespace);
        assert_eq!(parsed.output.expectation_met, Some(true));
        assert!(parsed.ttft.is_some());
        assert_eq!(parsed.prompt_tokens, Some(4));
        assert_eq!(parsed.completion_tokens, Some(3));
        assert_eq!(parsed.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn reasoning_starts_ttft_without_satisfying_the_content_expectation() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"watchman-ok\"},",
            "\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ready\"},",
            "\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,",
            "\"completion_tokens\":8}}\n\n",
            "data: [DONE]\n\n"
        );

        let parsed = parse_stream_reader(
            std::io::Cursor::new(body),
            Instant::now(),
            body.len(),
            Some("ready"),
        )
        .unwrap();

        assert!(parsed.ttft.is_some());
        assert!(parsed.output.has_non_whitespace);
        assert_eq!(parsed.output.expectation_met, Some(true));
        assert_eq!(parsed.completion_tokens, Some(8));

        let reasoning_only = body.replace("content\":\"ready", "content\":\"wrong");
        let parsed = parse_stream_reader(
            std::io::Cursor::new(&reasoning_only),
            Instant::now(),
            reasoning_only.len(),
            Some("watchman-ok"),
        )
        .unwrap();
        assert_eq!(parsed.output.expectation_met, Some(false));
    }

    #[test]
    fn protocol_errors_do_not_echo_untrusted_response_content() {
        let private = "generated-private-output";
        let body = format!("data: {private}\n\n");
        let error = parse_stream_reader(
            std::io::Cursor::new(&body),
            Instant::now(),
            body.len(),
            None,
        )
        .unwrap_err();

        assert!(!error.message.contains(private));
        assert_eq!(error.stage, CanaryFailureStage::Protocol);
    }

    #[test]
    fn non_stream_content_arrays_and_usage_are_supported() {
        let body = br#"{
          "model":"untrusted-response-model",
          "choices":[{"message":{"content":[{"type":"text","text":"ready"}]},
                      "finish_reason":"stop"}],
          "usage":{"prompt_tokens":2,"completion_tokens":1}
        }"#;
        let parsed =
            parse_json_reader(std::io::Cursor::new(body), body.len(), Some("ready")).unwrap();

        assert!(parsed.output.has_non_whitespace);
        assert_eq!(parsed.output.expectation_met, Some(true));
        assert_eq!(parsed.prompt_tokens, Some(2));
        assert_eq!(parsed.completion_tokens, Some(1));
        assert_eq!(parsed.finish_reason.as_deref(), Some("stop"));
        assert!(parsed.ttft.is_none());
    }

    #[test]
    fn parsers_fail_when_the_retained_body_exceeds_its_bound() {
        let json = br#"{"choices":[]}"#;
        let json_error = parse_json_reader(std::io::Cursor::new(json), 4, None).unwrap_err();
        assert!(json_error.message.contains("size limit"));

        let stream = b"data: [DONE]\n\n";
        let stream_error =
            parse_stream_reader(std::io::Cursor::new(stream), Instant::now(), 4, None).unwrap_err();
        assert!(stream_error.message.contains("size limit"));
    }

    #[test]
    fn rejects_a_truncated_stream_even_when_expected_output_arrived() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"gpu-watchman-ok\"}}]}\n\n";
        let error = parse_stream_reader(
            std::io::Cursor::new(body),
            Instant::now(),
            body.len(),
            Some("gpu-watchman-ok"),
        )
        .unwrap_err();

        assert_eq!(error.stage, CanaryFailureStage::Protocol);
        assert!(error.message.contains("terminal"));
    }

    #[test]
    fn done_sentinel_finishes_without_waiting_for_trailing_bytes() {
        let body = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ready\"},",
            "\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
            "data: definitely-not-json\n\n"
        );
        let parsed = parse_stream_reader(
            std::io::Cursor::new(body),
            Instant::now(),
            body.len(),
            Some("ready"),
        )
        .unwrap();

        assert_eq!(parsed.output.expectation_met, Some(true));
    }

    #[test]
    fn choices_cannot_be_concatenated_into_an_expectation_match() {
        let body = concat!(
            "data: {\"choices\":[",
            "{\"index\":0,\"delta\":{\"content\":\"gpu-\"}},",
            "{\"index\":1,\"delta\":{\"content\":\"watchman-ok\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let error = parse_stream_reader(
            std::io::Cursor::new(body),
            Instant::now(),
            body.len(),
            Some("gpu-watchman-ok"),
        )
        .unwrap_err();

        assert!(error.message.contains("exactly one"));
    }

    #[test]
    fn low_level_client_rejects_oversized_response_limits() {
        let result = OpenAiClient::new(OpenAiClientOptions {
            base_url: "http://127.0.0.1:8000/v1".to_owned(),
            model: "model".to_owned(),
            api_key: None,
            prompt: "prompt".to_owned(),
            expectation: None,
            max_tokens: 1,
            timeout: Duration::from_secs(1),
            max_body_bytes: MAX_BODY_BYTES + 1,
            stream: true,
            allow_insecure_http: false,
        });

        assert!(result.is_err());
    }

    #[test]
    fn memory_budget_counts_request_escaping_response_parsing_and_matcher_state() {
        let request = request_body_memory_budget(10, 20).unwrap();
        assert_eq!(request, 2 * (512 + 6 * 30));

        let expectation_bytes = 10;
        let matcher = u64::try_from(expectation_bytes).unwrap()
            * (1 + u64::try_from(std::mem::size_of::<usize>()).unwrap());
        let streamed = worker_memory_budget(request, 100, expectation_bytes, true).unwrap();
        let json = worker_memory_budget(request, 100, expectation_bytes, false).unwrap();

        assert_eq!(
            streamed,
            request + 101 * (JSON_VALUE_MEMORY_MULTIPLIER + 4) + matcher + 1024 * 1024
        );
        assert_eq!(
            json,
            request + 101 * (JSON_VALUE_MEMORY_MULTIPLIER + 2) + matcher + 1024 * 1024
        );
    }

    #[test]
    fn memory_budget_arithmetic_fails_closed_on_overflow() {
        assert!(worker_memory_budget(u64::MAX, usize::MAX, usize::MAX, true).is_none());
        if usize::BITS == 64 {
            assert!(request_body_memory_budget(usize::MAX, usize::MAX).is_none());
        }
    }

    #[test]
    fn implausible_endpoint_usage_is_discarded_before_attempt_evidence() {
        let mut parsed = ParsedCompletion::new(None);
        parsed.output.push("ok");
        parsed.prompt_tokens = Some(u64::MAX);
        parsed.completion_tokens = Some(u64::MAX);

        let attempt = OpenAiClient::finish_attempt(0, 200, 1.0, Instant::now(), parsed, 128);

        assert!(attempt.success);
        assert_eq!(attempt.prompt_tokens, None);
        assert_eq!(attempt.completion_tokens, None);
        assert_eq!(attempt.output_tokens_per_second, None);
    }
}
