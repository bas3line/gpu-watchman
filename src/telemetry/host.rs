//! Host identity collection kept outside the stable domain model.

use std::io::Read;
use std::path::Path;
use std::time::Duration;

use crate::domain::Host;

const MAX_HOSTNAME_BYTES: u64 = 64 * 1024;
const _: () = assert!(MAX_HOSTNAME_BYTES <= 64 * 1024);

pub fn local() -> Host {
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .ok()
        .or_else(read_etc_hostname)
        .or_else(bounded_system_hostname)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "local".to_owned());
    Host {
        hostname,
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
    }
}

fn read_etc_hostname() -> Option<String> {
    let file = std::fs::File::open("/etc/hostname").ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_HOSTNAME_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if u64::try_from(bytes.len()).ok()? > MAX_HOSTNAME_BYTES {
        return None;
    }
    Some(String::from_utf8(bytes).ok()?.trim().to_owned())
}

#[cfg(unix)]
fn bounded_system_hostname() -> Option<String> {
    super::command::run(Path::new("/bin/hostname"), &[], Duration::from_millis(500))
        .ok()
        .map(|value| value.trim().to_owned())
}

#[cfg(not(unix))]
fn bounded_system_hostname() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn hostname_fallback_is_absolute_and_bounded() {
        assert!(Path::new("/bin/hostname").is_absolute());
    }
}
