//! Linux process, cgroup, container, and Kubernetes workload attribution.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

use crate::domain::GpuProcess;

const MAX_CMDLINE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_CGROUP_BYTES: u64 = 64 * 1024;
const MAX_STATUS_BYTES: u64 = 256 * 1024;
const MAX_PASSWD_BYTES: u64 = 1024 * 1024;

pub(super) fn local_users() -> HashMap<u32, String> {
    let Some(passwd) = read_bounded(Path::new("/etc/passwd"), MAX_PASSWD_BYTES) else {
        return HashMap::new();
    };
    String::from_utf8_lossy(&passwd)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(':');
            let name = fields.next()?;
            let _password = fields.next()?;
            let uid = fields.next()?.parse().ok()?;
            (!name.is_empty()).then(|| (uid, name.to_owned()))
        })
        .collect()
}

pub(super) fn enrich(process: &mut GpuProcess, local_users: &HashMap<u32, String>) {
    let base = PathBuf::from("/proc").join(process.pid.to_string());
    let Some(command) = read_bounded(&base.join("cmdline"), MAX_CMDLINE_BYTES) else {
        return;
    };
    String::from_utf8_lossy(&command)
        .replace('\0', " ")
        .trim()
        .clone_into(&mut process.command);
    String::from_utf8_lossy(
        &read_bounded(&base.join("cgroup"), MAX_CGROUP_BYTES).unwrap_or_default(),
    )
    .trim()
    .clone_into(&mut process.cgroup);
    if let Some(uid) = read_bounded(&base.join("status"), MAX_STATUS_BYTES)
        .as_deref()
        .and_then(|status| parse_uid(&String::from_utf8_lossy(status)))
    {
        process.owner = local_users
            .get(&uid)
            .cloned()
            .unwrap_or_else(|| uid.to_string());
    }
    process.container_id = container_id(&process.cgroup);
    process.kubernetes_pod_uid = kubernetes_pod_uid(&process.cgroup);
}

fn read_bounded(path: &Path, limit: u64) -> Option<Vec<u8>> {
    let file = File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes).ok()?;
    (u64::try_from(bytes.len()).ok()? <= limit).then_some(bytes)
}

fn parse_uid(status: &str) -> Option<u32> {
    status
        .lines()
        .find(|line| line.starts_with("Uid:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse().ok())
}

fn container_id(cgroup: &str) -> Option<String> {
    cgroup
        .split(['/', '-', ':', '.'])
        .find(|part| {
            part.len() >= 32 && part.chars().all(|character| character.is_ascii_hexdigit())
        })
        .map(str::to_owned)
}

fn kubernetes_pod_uid(cgroup: &str) -> Option<String> {
    cgroup.split(['/', ':']).find_map(|part| {
        let part = part.strip_prefix("pod")?;
        if part.len() >= 32 {
            Some(part.replace('_', "-"))
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_container_and_pod_identity() {
        let cgroup = "0::/kubepods.slice/pod12345678_1234_1234_1234_123456789abc/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.scope";
        assert_eq!(container_id(cgroup).unwrap().len(), 64);
        assert_eq!(
            kubernetes_pod_uid(cgroup).as_deref(),
            Some("12345678-1234-1234-1234-123456789abc")
        );
    }

    #[test]
    fn parses_only_local_passwd_records() {
        let line = "alice:x:1001:1001::/home/alice:/bin/sh";
        let mut fields = line.split(':');
        assert_eq!(fields.next(), Some("alice"));
        assert_eq!(
            fields.nth(1).and_then(|uid| uid.parse::<u32>().ok()),
            Some(1001)
        );
    }
}
