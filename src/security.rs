//! Shared transport, listener, and local-file security primitives.

use std::fs::{File, OpenOptions};
use std::io;
use std::net::SocketAddr;
use std::path::Path;

use url::{Host, Url};

pub(crate) fn url_is_loopback(url: &Url) -> bool {
    match url.host() {
        Some(Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(address)) => address.is_loopback(),
        Some(Host::Ipv6(address)) => address.is_loopback(),
        None => false,
    }
}

/// Validate a listen target without DNS access and classify its host scope.
pub(crate) fn listen_is_loopback(value: &str) -> Result<bool, &'static str> {
    let value = value.trim();
    if value.is_empty() || value.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err("listen address must contain one host and port");
    }
    if let Ok(address) = value.parse::<SocketAddr>() {
        return Ok(address.ip().is_loopback());
    }

    let url = Url::parse(&format!("http://{value}"))
        .map_err(|_| "listen address must use HOST:PORT or IP:PORT syntax")?;
    if url.host().is_none()
        || url.port().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("listen address must use HOST:PORT or IP:PORT syntax");
    }
    Ok(url_is_loopback(&url))
}

/// Open a local input without allowing FIFOs or devices to block during open.
///
/// Callers must still inspect the opened descriptor and reject non-regular
/// files. `no_follow` is appropriate for inputs whose final symlink is not part
/// of the supported contract.
#[cfg(unix)]
pub(crate) fn open_read_nonblocking(path: &Path, no_follow: bool) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    let mut options = OpenOptions::new();
    options.read(true);
    let mut flags = libc::O_NONBLOCK;
    if no_follow {
        flags |= libc::O_NOFOLLOW;
    }
    options.custom_flags(flags).open(path)
}

#[cfg(not(unix))]
pub(crate) fn open_read_nonblocking(path: &Path, _: bool) -> io::Result<File> {
    File::open(path)
}

/// Reject macOS ACL entries that grant permissions beyond the mode bits.
///
/// macOS commonly installs deny-only ACLs on home directories. Those are safe
/// and remain accepted; any `allow` entry is rejected because it may grant a
/// lower-privileged principal write or read access hidden from `mode()`.
#[cfg(target_os = "macos")]
pub(crate) fn reject_permissive_acl(path: &Path, label: &str) -> anyhow::Result<()> {
    use std::io::Read;
    use std::process::{Command, Stdio};

    const MAX_ACL_OUTPUT_BYTES: u64 = 64 * 1024;

    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut child = Command::new("/bin/ls")
        .args(["-lde"])
        .arg(&absolute)
        .env_clear()
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| anyhow::anyhow!("failed to inspect {label} ACL"))?;
    let mut output = Vec::new();
    {
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to inspect {label} ACL"))?;
        stdout
            .by_ref()
            .take(MAX_ACL_OUTPUT_BYTES + 1)
            .read_to_end(&mut output)
            .map_err(|_| anyhow::anyhow!("failed to inspect {label} ACL"))?;
    }
    if u64::try_from(output.len()).unwrap_or(u64::MAX) > MAX_ACL_OUTPUT_BYTES {
        let _ = child.kill();
        let _ = child.wait();
        anyhow::bail!("{label} ACL is too large to validate safely");
    }
    if !child
        .wait()
        .map_err(|_| anyhow::anyhow!("failed to inspect {label} ACL"))?
        .success()
    {
        anyhow::bail!("failed to inspect {label} ACL");
    }
    if output.split(|byte| *byte == b'\n').skip(1).any(|line| {
        line.windows(b" allow ".len())
            .any(|part| part == b" allow ")
    }) {
        anyhow::bail!("{label} {} has a permission-granting ACL", path.display());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)] // Preserve one fallible cross-platform call shape.
pub(crate) fn reject_permissive_acl(_: &Path, _: &str) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listener_scope_is_syntactic_and_dns_free() {
        for local in ["127.0.0.1:9400", "[::1]:9400", "localhost:9400"] {
            assert_eq!(listen_is_loopback(local), Ok(true));
        }
        for remote in ["0.0.0.0:9400", "[::]:9400", "observer.internal:9400"] {
            assert_eq!(listen_is_loopback(remote), Ok(false));
        }
        for invalid in ["", "127.0.0.1", "http://127.0.0.1:9400", "user@host:1"] {
            assert!(listen_is_loopback(invalid).is_err(), "accepted {invalid:?}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn permission_granting_acl_is_rejected_but_deny_only_home_acl_is_allowed() {
        use std::process::{Command, Stdio};

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("private-token");
        std::fs::write(&path, b"private").unwrap();
        let status = Command::new("/bin/chmod")
            .args(["+a", "everyone allow read"])
            .arg(&path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success());

        assert!(reject_permissive_acl(&path, "test file").is_err());
        assert!(reject_permissive_acl(directory.path(), "test directory").is_ok());
    }
}
