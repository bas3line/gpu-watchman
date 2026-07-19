//! Portable, private-by-default incident support bundles.

use std::path::Path;
use std::time::Duration;
use std::{io, io::Write as _};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use serde::Serialize;

use crate::analysis::Tracker;
use crate::domain::Report;
use crate::inference::ProbeOptions;
use crate::operations::doctor;
use crate::security::reject_permissive_acl;
use crate::telemetry::NvidiaCollector;
use crate::{CycleOptions, collect_cycle};

const MAX_BUNDLE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Serialize)]
pub struct SupportBundle {
    pub bundle_version: u32,
    pub generated_at: chrono::DateTime<Utc>,
    pub gpu_watchman_version: &'static str,
    pub report: Report,
    pub checks: Vec<doctor::Check>,
}

pub fn write(
    output: &Path,
    nvidia_smi: &Path,
    timeout: Duration,
    probes: &[String],
    probe_token: Option<String>,
    allow_insecure_http: bool,
    no_xid: bool,
) -> Result<()> {
    let collector = NvidiaCollector::new(nvidia_smi, timeout);
    let collector = if no_xid {
        collector.without_xid()
    } else {
        collector
    };
    let options = CycleOptions {
        probe_urls: probes.to_vec(),
        probe: ProbeOptions {
            timeout,
            bearer_token: probe_token,
            allow_insecure_http,
            ..ProbeOptions::default()
        },
        ..CycleOptions::default()
    };
    let report = collect_cycle(&collector, &mut Tracker::new(256), &options)?;
    let checks = doctor::from_report(nvidia_smi, timeout, &report);
    let bundle = SupportBundle {
        bundle_version: 1,
        generated_at: Utc::now(),
        gpu_watchman_version: env!("CARGO_PKG_VERSION"),
        report,
        checks,
    };
    let mut open = std::fs::OpenOptions::new();
    open.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open.mode(0o600);
    }
    let file = open
        .open(output)
        .with_context(|| format!("create support bundle {}", output.display()))?;
    let result = (|| {
        validate_private_bundle(output, &file)?;
        let mut writer = BoundedWriter::new(file, MAX_BUNDLE_BYTES);
        if let Err(error) = serde_json::to_writer_pretty(&mut writer, &bundle) {
            if writer.exceeded {
                bail!("support bundle exceeds the 64 MiB limit");
            }
            return Err(error).context("encode support bundle");
        }
        writer.flush().context("flush support bundle")?;
        writer.file.sync_all().context("sync support bundle")?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(output);
        return Err(error);
    }
    Ok(())
}

struct BoundedWriter {
    file: std::fs::File,
    written: u64,
    limit: u64,
    exceeded: bool,
}

impl BoundedWriter {
    const fn new(file: std::fs::File, limit: u64) -> Self {
        Self {
            file,
            written: 0,
            limit,
            exceeded: false,
        }
    }
}

impl io::Write for BoundedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let length = u64::try_from(buffer.len()).unwrap_or(u64::MAX);
        if length > self.limit.saturating_sub(self.written) {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "support bundle exceeds the 64 MiB limit",
            ));
        }
        let written = self.file.write(buffer)?;
        self.written = self
            .written
            .saturating_add(u64::try_from(written).unwrap_or(u64::MAX));
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(unix)]
fn validate_private_bundle(output: &Path, file: &std::fs::File) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let opened = file
        .metadata()
        .with_context(|| format!("inspect support bundle {}", output.display()))?;
    let path = std::fs::metadata(output)
        .with_context(|| format!("inspect support bundle path {}", output.display()))?;
    if !opened.is_file() || opened.dev() != path.dev() || opened.ino() != path.ino() {
        bail!("support bundle path changed while it was being created");
    }
    if opened.permissions().mode() & 0o077 != 0 {
        bail!("support bundle must not grant group or other permissions");
    }
    let current_uid = uzers::get_current_uid();
    if !matches!(opened.uid(), 0) && opened.uid() != current_uid {
        bail!("support bundle must be owned by the current user or root");
    }
    reject_permissive_acl(output, "support bundle")?;
    Ok(())
}

#[cfg(not(unix))]
fn validate_private_bundle(_: &Path, _: &std::fs::File) -> Result<()> {
    Ok(())
}
