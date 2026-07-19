//! Stable data contract for active OpenAI-compatible inference checks.

use chrono::{DateTime, Utc};
use std::fmt;
use std::marker::PhantomData;

use serde::de::{Error as _, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// Current version of the standalone canary result contract.
pub const CANARY_VERSION: u32 = 2;
pub const DEFAULT_CANARY_WORKLOAD_ID: &str = "builtin-v1";
pub const MAX_CANARY_WORKLOAD_ID_BYTES: usize = 128;
pub const MAX_CANARY_IDENTITY_BYTES: usize = 64 << 10;
pub const MAX_CANARY_FAILURE_BYTES: usize = 512;
pub const MAX_CANARY_RETAINED_STRING_BYTES: usize = 512;
pub const MAX_CANARY_ATTEMPTS: usize = 10_000;
pub const MAX_CANARY_GATES: usize = 5;

/// Whether a workload identity is bounded and safe to retain in reports.
pub fn valid_canary_workload_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_CANARY_WORKLOAD_ID_BYTES
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric()
                || (index > 0 && matches!(byte, b'.' | b'_' | b':' | b'/' | b'-'))
        })
}

/// Overall outcome after request validation and configured gates.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CanaryStatus {
    Pass,
    #[default]
    Fail,
}

impl CanaryStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }
}

/// Failure stage for an individual inference request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CanaryFailureStage {
    Transport,
    Http,
    Protocol,
    EmptyOutput,
    Expectation,
}

/// Privacy-safe description of an individual request failure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanaryFailure {
    pub stage: CanaryFailureStage,
    #[serde(deserialize_with = "deserialize_failure_string")]
    pub message: String,
}

/// Redacted target identity. Request prompts, output, and credentials are never included.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanaryTarget {
    #[serde(deserialize_with = "deserialize_identity_string")]
    pub url: String,
    #[serde(deserialize_with = "deserialize_retained_string")]
    pub route: String,
    #[serde(deserialize_with = "deserialize_identity_string")]
    pub model: String,
    pub stream: bool,
}

/// Workload shape used for this bounded canary run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanaryPlan {
    pub count: u32,
    pub concurrency: u32,
    pub max_tokens: u32,
    pub timeout_ms: u64,
    pub response_limit_bytes: usize,
}

/// Exact distribution over the available finite samples.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CanaryDistribution {
    pub samples: usize,
    pub min: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

/// Aggregated result across all attempted requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CanarySummary {
    pub attempted: u32,
    pub succeeded: u32,
    pub failed: u32,
    pub success_percent: f64,
    pub achieved_requests_per_second: f64,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers_ms: Option<CanaryDistribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<CanaryDistribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e_ms: Option<CanaryDistribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_per_second: Option<CanaryDistribution>,
}

/// Privacy-safe evidence for the policy that produced a canary status.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CanaryPolicy {
    pub min_success_percent: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ttft_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_e2e_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_output_tokens_per_second: Option<f64>,
    pub expectation_configured: bool,
}

/// One evaluated service-level gate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanaryGate {
    #[serde(deserialize_with = "deserialize_retained_string")]
    pub name: String,
    #[serde(deserialize_with = "deserialize_retained_string")]
    pub operator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    pub threshold: f64,
    pub passed: bool,
    #[serde(
        default,
        deserialize_with = "deserialize_failure_string",
        skip_serializing_if = "String::is_empty"
    )]
    pub detail: String,
}

/// One request result. It intentionally contains no prompt or generated content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanaryAttempt {
    pub index: u32,
    pub success: bool,
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub status_code: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e2e_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_per_second: Option<f64>,
    #[serde(
        default,
        deserialize_with = "deserialize_identity_string",
        skip_serializing_if = "String::is_empty"
    )]
    pub model: String,
    #[serde(
        default,
        deserialize_with = "deserialize_retained_string",
        skip_serializing_if = "String::is_empty"
    )]
    pub finish_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expectation_met: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<CanaryFailure>,
}

impl CanaryAttempt {
    pub fn failed(index: u32, stage: CanaryFailureStage, message: impl Into<String>) -> Self {
        Self {
            index,
            success: false,
            status_code: 0,
            headers_ms: None,
            ttft_ms: None,
            e2e_ms: None,
            prompt_tokens: None,
            completion_tokens: None,
            output_tokens_per_second: None,
            model: String::new(),
            finish_reason: String::new(),
            expectation_met: None,
            failure: Some(CanaryFailure {
                stage,
                message: message.into(),
            }),
        }
    }
}

/// Complete standalone canary result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanaryReport {
    pub canary_version: u32,
    pub started_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub status: CanaryStatus,
    /// Operator-supplied, non-secret identity for the exact synthetic workload.
    ///
    /// Version-1 reports deserialize this as an empty value and are therefore
    /// intentionally incompatible with rollout comparison.
    #[serde(default, deserialize_with = "deserialize_workload_id")]
    pub workload_id: String,
    pub target: CanaryTarget,
    pub plan: CanaryPlan,
    /// Missing on version-1 input and therefore never rollout-compatible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<CanaryPolicy>,
    pub summary: CanarySummary,
    #[serde(
        default,
        deserialize_with = "deserialize_gates",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub gates: Vec<CanaryGate>,
    #[serde(deserialize_with = "deserialize_attempts")]
    pub attempts: Vec<CanaryAttempt>,
}

fn deserialize_workload_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CANARY_WORKLOAD_ID_BYTES>(deserializer)
}

fn deserialize_identity_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CANARY_IDENTITY_BYTES>(deserializer)
}

fn deserialize_failure_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CANARY_FAILURE_BYTES>(deserializer)
}

fn deserialize_retained_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_string::<D, MAX_CANARY_RETAINED_STRING_BYTES>(deserializer)
}

fn deserialize_bounded_string<'de, D, const MAX: usize>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundedStringVisitor<const MAX: usize>;

    impl<const MAX: usize> Visitor<'_> for BoundedStringVisitor<MAX> {
        type Value = String;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a UTF-8 string no longer than {MAX} bytes")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if value.len() > MAX {
                return Err(E::custom("string exceeds its canary report safety limit"));
            }
            Ok(value.to_owned())
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if value.len() > MAX {
                return Err(E::custom("string exceeds its canary report safety limit"));
            }
            Ok(value)
        }
    }

    deserializer.deserialize_string(BoundedStringVisitor::<MAX>)
}

fn deserialize_gates<'de, D>(deserializer: D) -> Result<Vec<CanaryGate>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, CanaryGate, MAX_CANARY_GATES>(deserializer)
}

fn deserialize_attempts<'de, D>(deserializer: D) -> Result<Vec<CanaryAttempt>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, CanaryAttempt, MAX_CANARY_ATTEMPTS>(deserializer)
}

fn deserialize_bounded_vec<'de, D, T, const MAX: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct BoundedVecVisitor<T, const MAX: usize>(PhantomData<T>);

    impl<'de, T, const MAX: usize> Visitor<'de> for BoundedVecVisitor<T, MAX>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "a sequence with no more than {MAX} entries")
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            if sequence.size_hint().is_some_and(|size| size > MAX) {
                return Err(A::Error::custom(
                    "sequence exceeds its canary report safety limit",
                ));
            }
            let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX));
            while let Some(value) = sequence.next_element()? {
                if values.len() == MAX {
                    return Err(A::Error::custom(
                        "sequence exceeds its canary report safety limit",
                    ));
                }
                values.push(value);
            }
            Ok(values)
        }
    }

    deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX>(PhantomData))
}

#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_zero_u16(value: &u16) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_contract_omits_absent_attempt_fields() {
        let attempt = CanaryAttempt::failed(2, CanaryFailureStage::Transport, "request failed");
        let encoded = serde_json::to_value(attempt).unwrap();

        assert_eq!(encoded["index"], 2);
        assert_eq!(encoded["failure"]["stage"], "transport");
        assert!(encoded.get("status_code").is_none());
        assert!(encoded.get("ttft_ms").is_none());
        assert_eq!(CANARY_VERSION, 2);
    }

    #[test]
    fn version_one_reports_default_the_missing_workload_identity() {
        let report: CanaryReport = serde_json::from_value(serde_json::json!({
            "canary_version": 1,
            "started_at": "2026-07-18T00:00:00Z",
            "duration_ms": 1,
            "status": "pass",
            "target": {
                "url": "http://localhost/v1/chat/completions",
                "route": "chat_completions",
                "model": "model",
                "stream": true
            },
            "plan": {
                "count": 1,
                "concurrency": 1,
                "max_tokens": 16,
                "timeout_ms": 1000,
                "response_limit_bytes": 1024
            },
            "summary": {
                "attempted": 1,
                "succeeded": 1,
                "failed": 0,
                "success_percent": 100.0,
                "achieved_requests_per_second": 1.0,
                "prompt_tokens_total": 1,
                "completion_tokens_total": 1
            },
            "attempts": []
        }))
        .unwrap();

        assert!(report.workload_id.is_empty());
    }

    #[test]
    fn workload_identity_is_bounded_visible_and_non_secret_by_contract() {
        for valid in ["builtin-v1", "team/chat_v2", "model:smoke.3"] {
            assert!(valid_canary_workload_id(valid), "{valid}");
        }
        for invalid in ["", "-leading", "contains whitespace", "line\nbreak", "🔒"] {
            assert!(!valid_canary_workload_id(invalid), "{invalid:?}");
        }
        assert!(!valid_canary_workload_id(
            &"x".repeat(MAX_CANARY_WORKLOAD_ID_BYTES + 1)
        ));
    }
}
