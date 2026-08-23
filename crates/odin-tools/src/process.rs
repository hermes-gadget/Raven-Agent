//! Bounded subprocess execution shared by command-backed tools.

use std::io;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};

/// Default retained output per stream for command-backed tools.
pub(crate) const DEFAULT_MAX_OUTPUT_BYTES: usize = 1024 * 1024;

/// Output retained from a completed subprocess.
pub(crate) struct BoundedOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
}

/// Errors that can occur while running a subprocess.
pub(crate) enum ProcessError {
    Io(io::Error),
    Timeout,
}

impl From<io::Error> for ProcessError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

async fn read_bounded<R: AsyncRead + Unpin>(
    mut reader: R,
    max_bytes: usize,
) -> io::Result<(Vec<u8>, bool)> {
    let mut retained = Vec::with_capacity(max_bytes.min(8192));
    let mut buffer = [0_u8; 8192];
    let mut truncated = false;

    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }

        let remaining = max_bytes.saturating_sub(retained.len());
        let keep = remaining.min(read);
        retained.extend_from_slice(&buffer[..keep]);
        if keep < read {
            truncated = true;
        }
    }

    Ok((retained, truncated))
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    // Put the command in its own group so a timeout can terminate shell
    // descendants as well as the direct child.
    command.as_std_mut().process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

async fn terminate(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // The child is the process-group leader (see configure_process_group).
        // Ignore ESRCH: the child may have exited while the timeout fired.
        unsafe {
            libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
        }
    }

    let _ = child.kill().await;
    let _ = child.wait().await;
}

/// Spawn a command, drain both output pipes, and enforce a hard timeout and
/// per-stream retention limit.
pub(crate) async fn run_command(
    mut command: Command,
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<BoundedOutput, ProcessError> {
    configure_process_group(&mut command);
    command.kill_on_drop(true);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("subprocess stdout was not piped"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("subprocess stderr was not piped"))?;
    let max_output_bytes = max_output_bytes.max(1);

    let result = tokio::time::timeout(timeout, async {
        let (stdout, stderr) = tokio::join!(
            read_bounded(stdout, max_output_bytes),
            read_bounded(stderr, max_output_bytes)
        );
        let status = child.wait().await?;
        let (stdout, stdout_truncated) = stdout?;
        let (stderr, stderr_truncated) = stderr?;
        Ok::<_, io::Error>(BoundedOutput {
            status,
            stdout,
            stderr,
            stdout_truncated,
            stderr_truncated,
        })
    })
    .await;

    match result {
        Ok(output) => output.map_err(ProcessError::Io),
        Err(_) => {
            terminate(&mut child).await;
            Err(ProcessError::Timeout)
        }
    }
}
