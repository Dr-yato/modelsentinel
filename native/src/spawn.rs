//! Cross-platform child execution with a wall-clock timeout, plus a PATH
//! lookup that works the same way on Linux, macOS and Windows.

use std::env;
use std::ffi::OsString;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct Outcome {
    pub stdout: String,
    pub stderr: String,
    /// `None` when the process was killed by a signal or by the timeout.
    pub status: Option<i32>,
    pub timed_out: bool,
}

/// Find an executable the way a shell would, using the *parent's* `PATH`.
/// The child gets a cleared environment, so the lookup happens up front.
pub fn find_program(name: &str) -> Option<PathBuf> {
    let direct = Path::new(name);
    if direct.components().count() > 1 || direct.is_absolute() {
        return direct.is_file().then(|| direct.to_path_buf());
    }
    let exts: Vec<OsString> = if cfg!(windows) {
        let raw = env::var("PATHEXT").unwrap_or_else(|_| ".EXE;.COM;.BAT;.CMD".into());
        std::iter::once(OsString::new())
            .chain(raw.split(';').filter(|e| !e.is_empty()).map(OsString::from))
            .collect()
    } else {
        vec![OsString::new()]
    };
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        for ext in &exts {
            let mut file = OsString::from(name);
            file.push(ext);
            let candidate = dir.join(&file);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Default interpreter names, in preference order, for this platform.
pub fn python_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python.exe", "python3.exe"]
    } else {
        &["python3", "python"]
    }
}

/// `SENTINEL_PYTHON` if set, else the first interpreter found on `PATH`.
pub fn resolve_python() -> Result<PathBuf, String> {
    if let Ok(chosen) = env::var("SENTINEL_PYTHON") {
        return find_program(&chosen)
            .ok_or_else(|| format!("SENTINEL_PYTHON={chosen} was not found"));
    }
    python_names()
        .iter()
        .find_map(|name| find_program(name))
        .ok_or_else(|| format!("no Python interpreter found on PATH (tried {:?}); set SENTINEL_PYTHON", python_names()))
}

fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = pipe {
            // Cap what we keep so a snippet printing forever cannot exhaust memory.
            let mut chunk = [0u8; 8192];
            while let Ok(n) = pipe.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if buf.len() < 1 << 20 {
                    buf.extend_from_slice(&chunk[..n]);
                }
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    })
}

/// Spawn `cmd` with piped output and wait at most `timeout`. Output is read on
/// background threads, so a chatty child cannot deadlock on a full pipe.
pub fn run_with_timeout(mut cmd: Command, timeout: Duration) -> io::Result<Outcome> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child: Child = cmd.spawn()?;
    let out = drain(child.stdout.take());
    let err = drain(child.stderr.take());
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait()? {
            break (status.code(), false);
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            break (None, true);
        }
        thread::sleep(Duration::from_millis(20));
    };
    Ok(Outcome {
        stdout: out.join().unwrap_or_default(),
        stderr: err.join().unwrap_or_default(),
        status,
        timed_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_program_is_none() {
        assert!(find_program("definitely-not-a-real-binary-sentinel").is_none());
    }

    #[test]
    fn python_runs_and_times_out() {
        let Ok(python) = resolve_python() else {
            eprintln!("skipping: no python on PATH");
            return;
        };
        let mut ok = Command::new(&python);
        ok.args(["-c", "print('hi' * 20000)"]);
        let done = run_with_timeout(ok, Duration::from_secs(20)).unwrap();
        assert_eq!(done.status, Some(0));
        assert!(!done.timed_out);
        assert_eq!(done.stdout.trim().len(), 40000);

        let mut slow = Command::new(&python);
        slow.args(["-c", "import time; time.sleep(30)"]);
        let killed = run_with_timeout(slow, Duration::from_millis(500)).unwrap();
        assert!(killed.timed_out);
        assert_eq!(killed.status, None);
    }
}
