//! Deadlock-safe child execution with a hard deadline and null stdin.

use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use wait_timeout::ChildExt;

const MAX_STDOUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_STDERR_BYTES: usize = 64 * 1024;
const GPU_WATCHMAN_ENVIRONMENT: &[&str] = &[
    "GPU_WATCHMAN_API_TOKEN",
    "GPU_WATCHMAN_API_TOKEN_FILE",
    "GPU_WATCHMAN_CONFIG",
    "GPU_WATCHMAN_INFERENCE_API_KEY",
    "GPU_WATCHMAN_INFERENCE_API_KEY_FILE",
    "GPU_WATCHMAN_INFERENCE_MODEL",
    "GPU_WATCHMAN_INFERENCE_URL",
    "GPU_WATCHMAN_NVIDIA_SMI",
    "GPU_WATCHMAN_PROBE_TOKEN",
    "GPU_WATCHMAN_PROBE_TOKEN_FILE",
    "GPU_WATCHMAN_PROFILE",
];

pub(super) fn run(command: &Path, args: &[&str], timeout: Duration) -> Result<String> {
    let started = Instant::now();
    let mut process = Command::new(command);
    process
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    scrub_gpu_watchman_environment(&mut process);
    configure_process_group(&mut process);
    let mut child = process
        .spawn()
        .with_context(|| format!("start {}", command.display()))?;

    // Waiting before draining can deadlock once either OS pipe buffer fills.
    let stdout_reader = spawn_pipe_reader(child.stdout.take(), MAX_STDOUT_BYTES);
    let stderr_reader = spawn_pipe_reader(child.stderr.take(), MAX_STDERR_BYTES);
    let status = match child.wait_timeout(remaining(timeout, started)) {
        Ok(status) => status,
        Err(error) => {
            terminate_and_reap(&mut child);
            return Err(error).with_context(|| format!("wait for {}", command.display()));
        }
    };
    let Some(status) = status else {
        terminate_and_reap(&mut child);
        bail!(
            "{} timed out after {}",
            command.display(),
            humantime::format_duration(timeout)
        );
    };
    let stdout = receive_pipe(
        &stdout_reader,
        "stdout",
        command,
        timeout,
        started,
        &mut child,
    )?;
    let stderr = receive_pipe(
        &stderr_reader,
        "stderr",
        command,
        timeout,
        started,
        &mut child,
    )?;
    if !status.success() {
        bail!(
            "{} exited with {status}: {}",
            command.display(),
            classify_diagnostic(&stderr.bytes, &stdout.bytes)
        );
    }
    if stdout.truncated {
        bail!(
            "{} stdout exceeds {} MiB",
            command.display(),
            MAX_STDOUT_BYTES / (1024 * 1024)
        );
    }
    String::from_utf8(stdout.bytes).context("command stdout is not valid UTF-8")
}

fn scrub_gpu_watchman_environment(command: &mut Command) {
    for name in GPU_WATCHMAN_ENVIRONMENT {
        command.env_remove(name);
    }
    // Future GPU Watchman variables inherited from the parent are stripped too,
    // so newly added credentials do not silently cross the process boundary.
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("GPU_WATCHMAN_") {
            command.env_remove(name);
        }
    }
}

fn classify_diagnostic(stderr: &[u8], stdout: &[u8]) -> &'static str {
    // Some nvidia-smi versions publish query errors on stdout. Inspect both
    // streams for a fixed classification, but never include their contents in
    // the returned error because wrappers may print private diagnostics.
    let diagnostic = format!(
        "{} {}",
        String::from_utf8_lossy(stderr),
        String::from_utf8_lossy(stdout)
    )
    .to_ascii_lowercase();
    if diagnostic.trim().is_empty() {
        "no diagnostic was published"
    } else if diagnostic.contains("permission denied")
        || diagnostic.contains("insufficient permission")
        || diagnostic.contains("not authorized")
    {
        "permission denied"
    } else if diagnostic.contains("no devices were found")
        || diagnostic.contains("no nvidia devices")
    {
        "no NVIDIA devices were found"
    } else if diagnostic.contains("not supported") || diagnostic.contains("unsupported") {
        "requested operation is not supported"
    } else if diagnostic.contains("invalid field")
        || diagnostic.contains("field is not a valid field")
        || diagnostic.contains("is not a valid field")
    {
        "driver query field is unavailable"
    } else if diagnostic.contains("not found") || diagnostic.contains("no such file") {
        "required command or resource was not found"
    } else {
        "diagnostic omitted; run the configured command locally for details"
    }
}

fn spawn_pipe_reader(
    pipe: Option<impl Read + Send + 'static>,
    max_bytes: usize,
) -> Receiver<Result<PipeOutput>> {
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = sender.send(read_pipe(pipe, max_bytes));
    });
    receiver
}

fn receive_pipe(
    receiver: &Receiver<Result<PipeOutput>>,
    name: &str,
    command: &Path,
    timeout: Duration,
    started: Instant,
    child: &mut Child,
) -> Result<PipeOutput> {
    match receiver.recv_timeout(remaining(timeout, started)) {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(error)) => {
            terminate_and_reap(child);
            Err(error)
        }
        Err(RecvTimeoutError::Timeout) => {
            terminate_and_reap(child);
            bail!(
                "{} timed out after {} while draining {name}",
                command.display(),
                humantime::format_duration(timeout)
            );
        }
        Err(RecvTimeoutError::Disconnected) => {
            terminate_and_reap(child);
            bail!("{name} reader terminated unexpectedly");
        }
    }
}

fn remaining(timeout: Duration, started: Instant) -> Duration {
    timeout.saturating_sub(started.elapsed())
}

fn terminate_and_reap(child: &mut Child) {
    terminate_process_group(child.id());
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt as _;

    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn terminate_process_group(process_group: u32) {
    // `std` can create a process group safely but does not expose killpg. Use
    // the platform kill utility, then retain Child::kill as a direct fallback.
    let target = format!("-{process_group}");
    let _ = Command::new("/bin/kill")
        .args(["-KILL", "--"])
        .arg(target)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(not(unix))]
fn terminate_process_group(_process_group: u32) {}

#[derive(Debug, Default, PartialEq, Eq)]
struct PipeOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

fn read_pipe(pipe: Option<impl Read>, max_bytes: usize) -> Result<PipeOutput> {
    let Some(mut pipe) = pipe else {
        return Ok(PipeOutput::default());
    };
    let mut output = PipeOutput {
        bytes: Vec::with_capacity(max_bytes.min(64 * 1024)),
        truncated: false,
    };
    let mut buffer = [0_u8; 8192];
    loop {
        let read = pipe.read(&mut buffer).context("read command pipe")?;
        if read == 0 {
            break;
        }
        let remaining = max_bytes.saturating_sub(output.bytes.len());
        let retained = remaining.min(read);
        output.bytes.extend_from_slice(&buffer[..retained]);
        output.truncated |= retained < read;
    }
    Ok(output)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn terminates_a_child_that_exceeds_its_deadline() {
        let error = run(
            Path::new("/bin/sh"),
            &["-c", "while :; do :; done"],
            Duration::from_millis(20),
        )
        .unwrap_err();

        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn deadline_includes_pipes_held_by_descendants_and_kills_the_group() {
        let directory = tempfile::tempdir().unwrap();
        let process_group_path = directory.path().join("process-group");
        let process_group_path = process_group_path.to_str().unwrap();
        let started = Instant::now();
        let error = run(
            Path::new("/bin/sh"),
            &[
                "-c",
                "echo $$ > \"$1\"; trap '' HUP; sleep 30 & exit 0",
                "gpu-watchman-test",
                process_group_path,
            ],
            Duration::from_millis(150),
        )
        .unwrap_err();

        assert!(error.to_string().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(3));
        let process_group = std::fs::read_to_string(process_group_path).unwrap();
        assert!(wait_for_process_group_exit(process_group.trim()));
    }

    #[test]
    fn pipe_reader_retains_a_bound_while_draining_the_rest() {
        let output = read_pipe(Some(std::io::Cursor::new(b"0123456789")), 4).unwrap();

        assert_eq!(output.bytes, b"0123");
        assert!(output.truncated);
    }

    #[test]
    fn failed_command_stderr_is_classified_without_publishing_content() {
        let private = "wrapper-credential-private";
        let error = run(
            Path::new("/bin/sh"),
            &["-c", &format!("printf '%s' '{private}' >&2; exit 7")],
            Duration::from_secs(1),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("diagnostic omitted"));
        assert!(!error.contains(private));
        assert_eq!(
            classify_diagnostic(b"permission denied: private", b""),
            "permission denied"
        );
    }

    #[test]
    fn failed_command_stdout_is_classified_without_publishing_content() {
        let private = "Field \"private.query.field\" is not a valid field to query.";
        let error = run(
            Path::new("/bin/sh"),
            &["-c", &format!("printf '%s' '{private}'; exit 2")],
            Duration::from_secs(1),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("driver query field is unavailable"));
        assert!(!error.contains("private.query.field"));
    }

    #[test]
    fn child_environment_omits_gpu_watchman_configuration_and_credentials() {
        let mut command = Command::new("/usr/bin/env");
        command
            .env("GPU_WATCHMAN_API_TOKEN", "api-secret")
            .env("GPU_WATCHMAN_PROBE_TOKEN_FILE", "/private/probe-token")
            .env("GPU_WATCHMAN_INFERENCE_API_KEY", "inference-secret")
            .stdout(Stdio::piped());
        scrub_gpu_watchman_environment(&mut command);

        let output = command.output().unwrap();
        assert!(output.status.success());
        let output = String::from_utf8(output.stdout).unwrap();
        assert!(!output.contains("GPU_WATCHMAN_"));
        assert!(!output.contains("api-secret"));
        assert!(!output.contains("inference-secret"));
        assert!(!output.contains("/private/probe-token"));
    }

    fn wait_for_process_group_exit(process_group: &str) -> bool {
        let target = format!("-{process_group}");
        for _ in 0..50 {
            let exists = Command::new("/bin/kill")
                .args(["-0", "--"])
                .arg(&target)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success());
            if !exists {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }
}
