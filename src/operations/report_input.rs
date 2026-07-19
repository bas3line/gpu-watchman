//! Bounded, descriptor-backed input for saved JSON or final-record NDJSON evidence.

use std::io::{ErrorKind, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;

use crate::security::open_read_nonblocking;

/// Load one typed JSON value or the final non-empty NDJSON record.
///
/// Decode failures retain only line and column locations. They never echo a
/// source excerpt or an attacker-controlled unknown enum/string value.
pub(crate) fn load_json_or_final_ndjson<T: DeserializeOwned>(
    path: &Path,
    label: &'static str,
    maximum_bytes: u64,
) -> Result<T> {
    let read_limit = maximum_bytes
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("{label} safety limit is invalid"))?;
    let mut file = open_read_nonblocking(path, false)
        .with_context(|| format!("open {label} {}", path.display()))?;
    let initial_metadata = file
        .metadata()
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    if !initial_metadata.is_file() {
        bail!("{label} {} is not a regular file", path.display());
    }
    if initial_metadata.len() > maximum_bytes {
        return too_large(path, label, maximum_bytes);
    }

    let capacity = usize::try_from(initial_metadata.len().min(maximum_bytes)).unwrap_or_default();
    let bytes = read_bounded(&mut file, capacity, maximum_bytes, read_limit)
        .with_context(|| format!("read {label} {}", path.display()))?;
    let Some(bytes) = bytes else {
        return too_large(path, label, maximum_bytes);
    };

    let final_metadata = file
        .metadata()
        .with_context(|| format!("reinspect {label} {}", path.display()))?;
    let bytes_read = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let modified_changed = initial_metadata
        .modified()
        .ok()
        .zip(final_metadata.modified().ok())
        .is_some_and(|(before, after)| before != after);
    if final_metadata.len() != initial_metadata.len()
        || bytes_read != initial_metadata.len()
        || modified_changed
    {
        bail!("{label} {} changed while it was being read", path.display());
    }

    let body = std::str::from_utf8(&bytes)
        .with_context(|| format!("{label} {} is not UTF-8", path.display()))?;
    match serde_json::from_str(body) {
        Ok(value) => Ok(value),
        Err(full_error) => {
            let Some(last) = body.lines().rev().find(|line| !line.trim().is_empty()) else {
                bail!("{label} {} is empty", path.display());
            };
            serde_json::from_str(last).map_err(|last_error| {
                anyhow::anyhow!(
                    "decode {label} {} as JSON or final NDJSON record (JSON line {}, column {}; final record line {}, column {})",
                    path.display(),
                    full_error.line(),
                    full_error.column(),
                    last_error.line(),
                    last_error.column()
                )
            })
        }
    }
}

fn read_bounded(
    reader: &mut impl Read,
    capacity: usize,
    maximum_bytes: u64,
    read_limit: u64,
) -> std::io::Result<Option<Vec<u8>>> {
    if read_limit <= maximum_bytes {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "report read limit overflowed",
        ));
    }
    let mut bytes = Vec::with_capacity(capacity);
    Read::by_ref(reader)
        .take(read_limit)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > maximum_bytes {
        Ok(None)
    } else {
        Ok(Some(bytes))
    }
}

fn too_large<T>(path: &Path, label: &'static str, maximum_bytes: u64) -> Result<T> {
    bail!(
        "{label} {} is larger than the {} MiB safety limit",
        path.display(),
        maximum_bytes / (1024 * 1024)
    )
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write as _};

    use serde::Deserialize;
    use tempfile::{NamedTempFile, tempdir};

    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Fixture {
        value: u32,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum FixedValue {
        Allowed,
    }

    #[derive(Debug, Deserialize)]
    struct FixedFixture {
        #[allow(dead_code)]
        kind: FixedValue,
    }

    #[test]
    fn loads_pretty_json_and_the_final_non_empty_ndjson_record() {
        let mut pretty = NamedTempFile::new().unwrap();
        write!(pretty, "{{\n  \"value\": 7\n}}\n").unwrap();
        assert_eq!(
            load_json_or_final_ndjson::<Fixture>(pretty.path(), "fixture", 1_024).unwrap(),
            Fixture { value: 7 }
        );

        let mut ndjson = NamedTempFile::new().unwrap();
        writeln!(ndjson, "{{\"value\":1}}").unwrap();
        writeln!(ndjson, "{{\"value\":2}}").unwrap();
        writeln!(ndjson).unwrap();
        assert_eq!(
            load_json_or_final_ndjson::<Fixture>(ndjson.path(), "fixture", 1_024).unwrap(),
            Fixture { value: 2 }
        );
    }

    #[test]
    fn accepts_a_value_at_the_exact_byte_limit() {
        let mut file = NamedTempFile::new().unwrap();
        let body = b"{\"value\":7}";
        file.write_all(body).unwrap();

        assert_eq!(
            load_json_or_final_ndjson::<Fixture>(
                file.path(),
                "fixture",
                u64::try_from(body.len()).unwrap()
            )
            .unwrap(),
            Fixture { value: 7 }
        );
    }

    #[test]
    fn rejects_non_regular_empty_and_invalid_utf8_inputs() {
        let directory = tempdir().unwrap();
        assert!(load_json_or_final_ndjson::<Fixture>(directory.path(), "fixture", 1_024).is_err());

        let empty = NamedTempFile::new().unwrap();
        let error = load_json_or_final_ndjson::<Fixture>(empty.path(), "fixture", 1_024)
            .unwrap_err()
            .to_string();
        assert!(error.contains("empty"));

        let mut whitespace = NamedTempFile::new().unwrap();
        whitespace.write_all(b" \n\t\r\n").unwrap();
        let error = load_json_or_final_ndjson::<Fixture>(whitespace.path(), "fixture", 1_024)
            .unwrap_err()
            .to_string();
        assert!(error.contains("empty"));

        let mut invalid_utf8 = NamedTempFile::new().unwrap();
        invalid_utf8.write_all(&[0xff, 0xfe]).unwrap();
        let error = load_json_or_final_ndjson::<Fixture>(invalid_utf8.path(), "fixture", 1_024)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not UTF-8"));
        assert!(!error.contains('\u{fffd}'));
    }

    #[test]
    fn rejects_metadata_and_post_admission_size_overflow() {
        let mut oversized = NamedTempFile::new().unwrap();
        oversized.write_all(b"123456789").unwrap();
        let error = load_json_or_final_ndjson::<Fixture>(oversized.path(), "fixture", 8)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safety limit"));

        // A reader yielding more bytes than its admitted capacity models a
        // regular file growing after the initial metadata inspection.
        let mut growing = Cursor::new(b"123456789".to_vec());
        let result = read_bounded(&mut growing, 8, 8, 9).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn decode_error_does_not_echo_an_attacker_controlled_value() {
        let secret = "private-invalid-value-that-must-not-be-echoed";
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{{\"value\":\"{secret}\"}}").unwrap();

        let error = load_json_or_final_ndjson::<Fixture>(file.path(), "fixture", 1_024)
            .unwrap_err()
            .to_string();

        assert!(error.contains("line"));
        assert!(!error.contains(secret));
    }

    #[test]
    fn unknown_enum_and_final_record_errors_are_location_only() {
        let secret = "private-unknown-variant-that-must-not-be-echoed";
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{{\"kind\":\"allowed\"}}").unwrap();
        writeln!(file, "{{\"kind\":\"{secret}\"}}").unwrap();

        let error = load_json_or_final_ndjson::<FixedFixture>(file.path(), "fixture", 1_024)
            .unwrap_err()
            .to_string();

        assert!(error.contains("JSON line"));
        assert!(error.contains("final record line"));
        assert!(!error.contains(secret));
        assert!(!error.contains("unknown variant"));
    }
}
