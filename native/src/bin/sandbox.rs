//! Run one generated snippet under a cleared environment, with canary secrets,
//! a seccomp network deny-list, and (when the kernel allows) a Landlock policy.
//!
//! macOS and Windows do not have this backend. The Python CLI falls back to
//! Docker (`--network none`) there when Docker is available.

use sentinel_native::restrict::{
    apply_landlock, apply_seccomp, close_fds, last_errno, network_filter, open_path, set_no_new_privs,
    LandlockFds, SockFilter,
};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const RUNNER: &str = include_str!("../sandbox_runner.py");

struct Args {
    code: PathBuf,
    timeout: Duration,
    docker: bool,
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            emit_error(&message);
            std::process::exit(2);
        }
    };
    let result = if args.docker {
        run_docker(&args)
    } else {
        run_native(&args)
    };
    match result {
        Ok(value) => println!("{value}"),
        Err(message) => {
            emit_error(&message);
            std::process::exit(1);
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut code = None;
    let mut timeout = 5u64;
    let mut docker = false;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--code" => {
                code = Some(PathBuf::from(iter.next().ok_or("missing --code value")?));
            }
            "--timeout" => {
                let raw = iter.next().ok_or("missing --timeout value")?;
                timeout = raw.parse::<u64>().map_err(|_| "timeout must be seconds")?;
            }
            "--docker" => docker = true,
            "-h" | "--help" => {
                println!("usage: sentinel-sandbox --code FILE [--timeout SECONDS] [--docker]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let code = code.ok_or("usage: sentinel-sandbox --code FILE [--timeout SECONDS]")?;
    if !code.is_file() {
        return Err(format!("code file not found: {}", code.display()));
    }
    Ok(Args {
        code,
        timeout: Duration::from_secs(timeout.max(1)),
        docker,
    })
}

fn run_native(args: &Args) -> Result<String, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let jail = env::temp_dir().join(format!("modelsentinel-jail-{stamp}-{}", std::process::id()));
    let home = jail.join("home");
    let work = jail.join("work");
    fs::create_dir_all(home.join(".ssh")).map_err(|e| e.to_string())?;
    fs::create_dir_all(home.join(".aws")).map_err(|e| e.to_string())?;
    fs::create_dir_all(&work).map_err(|e| e.to_string())?;

    let token = format!("{stamp:x}");
    let canaries = vec![
        write_canary(&home.join(".ssh/id_rsa"), &format!("SENTINEL_CANARY_SSH {token}\n"))?,
        write_canary(
            &home.join(".ssh/authorized_keys"),
            &format!("SENTINEL_CANARY_AUTHKEYS {token}\n"),
        )?,
        write_canary(
            &home.join(".aws/credentials"),
            &format!("[default]\naws_access_key_id = SENTINEL_CANARY_AWS_{token}\naws_secret_access_key = SENTINEL_CANARY_SECRET_{token}\n"),
        )?,
        write_canary(&home.join(".env"), &format!("SENTINEL_CANARY_DOTENV={token}\nAPI_TOKEN=SENTINEL_CANARY_TOKEN_{token}\n"))?,
    ];
    let snippet = work.join("snippet.py");
    fs::copy(&args.code, &snippet).map_err(|e| e.to_string())?;
    let runner = work.join("runner.py");
    fs::write(&runner, RUNNER).map_err(|e| e.to_string())?;
    let events = work.join("events.jsonl");
    let outside = env::temp_dir().join(format!("modelsentinel-outside-{stamp}.txt"));
    fs::write(&outside, b"outside-jail-marker\n").map_err(|e| e.to_string())?;

    let python = env::var("SENTINEL_PYTHON").unwrap_or_else(|_| "python3".into());
    let canary_env = canaries
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(":");

    let filter = network_filter();
    let filter: &'static [SockFilter] = Box::leak(filter.into_boxed_slice());
    let read_paths = ["/usr", "/lib", "/lib64", "/etc", "/dev", "/proc", "/bin", "/sys"];
    let mut fds = LandlockFds {
        read: Vec::new(),
        write: Vec::new(),
    };
    for path in read_paths {
        if Path::new(path).exists() {
            if let Some(fd) = open_path(path) {
                fds.read.push(fd);
            }
        }
    }
    if let Some(fd) = open_path(&jail.display().to_string()) {
        fds.write.push(fd);
    }

    let first = spawn_once(
        &python,
        &runner,
        &snippet,
        &events,
        &home,
        &work,
        &jail,
        &canary_env,
        &outside,
        args.timeout,
        filter,
        &fds,
        true,
    )?;
    let (body, stderr_tail, status, enforcement, used_landlock) = if python_failed_to_start(&first) {
        let second = spawn_once(
            &python,
            &runner,
            &snippet,
            &events,
            &home,
            &work,
            &jail,
            &canary_env,
            &outside,
            args.timeout,
            filter,
            &fds,
            false,
        )?;
        (
            second.stdout,
            second.stderr,
            second.status,
            second.enforcement,
            false,
        )
    } else {
        (
            first.stdout,
            first.stderr,
            first.status,
            first.enforcement,
            first.landlock_requested,
        )
    };

    close_fds(&fds);
    let _ = fs::remove_file(&outside);

    let restrict_status = fs::read_to_string(work.join("restrict.txt")).unwrap_or_default();
    let raw_events = fs::read_to_string(&events).unwrap_or_default();
    let mut events = Vec::new();
    for line in raw_events.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(value) => events.push(value),
            Err(_) => events.push(json!({"kind": "parse_error", "detail": line})),
        }
    }
    let (canary_read, outbound_attempt, filesystem, network) = summarise(&events);
    let mut enforcement = enforcement;
    if used_landlock && filesystem == "landlock-blocked" && !enforcement.iter().any(|e| e == "landlock") {
        enforcement.push("landlock".into());
    }
    if network == "eperm" && !enforcement.iter().any(|e| e == "seccomp") {
        enforcement.push("seccomp".into());
    }
    let value = json!({
        "enforcement": enforcement,
        "exit_code": status,
        "timed_out": status == -9,
        "canary_read": canary_read,
        "outbound_attempt": outbound_attempt,
        "network_probe": network,
        "filesystem_probe": filesystem,
        "events": events,
        "stdout": tail(&body, 2000),
        "stderr": tail(&stderr_tail, 2000),
        "canary_token": token,
        "jail": jail.display().to_string(),
        "restrict_status": restrict_status.trim(),
        "error": null,
    });
    // Leave the jail in place only if SENTINEL_KEEP_JAIL=1, so failures can be inspected.
    if env::var("SENTINEL_KEEP_JAIL").ok().as_deref() != Some("1") {
        let _ = fs::remove_dir_all(&jail);
    }
    Ok(value.to_string())
}

struct Spawned {
    stdout: String,
    stderr: String,
    status: i32,
    enforcement: Vec<String>,
    landlock_requested: bool,
}

fn spawn_once(
    python: &str,
    runner: &Path,
    snippet: &Path,
    events: &Path,
    home: &Path,
    work: &Path,
    jail: &Path,
    canary_env: &str,
    outside: &Path,
    timeout: Duration,
    filter: &'static [SockFilter],
    fds: &LandlockFds,
    landlock: bool,
) -> Result<Spawned, String> {
    let fds_read = fds.read.clone();
    let fds_write = fds.write.clone();
    let mut cmd = Command::new(python);
    cmd.arg(runner)
        .env_clear()
        .env("HOME", home)
        .env("TMPDIR", work)
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C.UTF-8")
        .env("LC_ALL", "C.UTF-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("SENTINEL_CODE", snippet)
        .env("SENTINEL_EVENTS", events)
        .env("SENTINEL_CANARIES", canary_env)
        .env("SENTINEL_OUTSIDE", outside)
        .current_dir(work)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let use_landlock = landlock && !fds_write.is_empty();
    let status_path = work.join("restrict.txt");
    let status_file = fs::File::create(&status_path).map_err(|e| e.to_string())?;
    use std::os::fd::IntoRawFd;
    let status_fd = status_file.into_raw_fd();
    unsafe {
        cmd.pre_exec(move || {
            // Landlock refuses to restrict a process that can still gain
            // privileges. Set no_new_privs before the ruleset, then seccomp.
            set_no_new_privs();
            if use_landlock {
                let borrowed = LandlockFds {
                    read: fds_read.clone(),
                    write: fds_write.clone(),
                };
                match apply_landlock(&borrowed) {
                    Ok(()) => {
                        let _ = libc::write(
                            status_fd,
                            b"landlock-ok\n".as_ptr() as *const libc::c_void,
                            12,
                        );
                    }
                    Err(_) => {
                        let err = last_errno();
                        let mut buf = [0u8; 32];
                        let text = b"landlock-err:";
                        buf[..text.len()].copy_from_slice(text);
                        let mut n = text.len();
                        let mut value = err.unsigned_abs();
                        let mut digits = [0u8; 16];
                        let mut d = 0;
                        if value == 0 {
                            digits[0] = b'0';
                            d = 1;
                        }
                        while value > 0 && d < digits.len() {
                            digits[d] = b'0' + (value % 10) as u8;
                            value /= 10;
                            d += 1;
                        }
                        while d > 0 {
                            d -= 1;
                            buf[n] = digits[d];
                            n += 1;
                        }
                        buf[n] = b'\n';
                        n += 1;
                        let _ = libc::write(status_fd, buf.as_ptr() as *const libc::c_void, n);
                    }
                }
            } else {
                let _ = libc::write(
                    status_fd,
                    b"landlock-skipped\n".as_ptr() as *const libc::c_void,
                    17,
                );
            }
            // A seccomp failure must not abort the run; the probe records it.
            let _ = apply_seccomp(filter);
            Ok(())
        });
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn python: {e}"))?;
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                let mut enforcement = vec!["audit-hook".into(), "cleared-env".into(), "canary-secrets".into()];
                if use_landlock {
                    enforcement.push("landlock-requested".into());
                }
                return Ok(Spawned {
                    stdout: String::new(),
                    stderr: "timed out".into(),
                    status: -9,
                    enforcement,
                    landlock_requested: use_landlock,
                });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(format!("wait: {e}")),
        }
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_string(&mut stdout);
    }
    if let Some(mut pipe) = child.stderr.take() {
        let _ = pipe.read_to_string(&mut stderr);
    }
    let mut enforcement = vec![
        "audit-hook".into(),
        "cleared-env".into(),
        "canary-secrets".into(),
        "seccomp-requested".into(),
    ];
    if use_landlock {
        enforcement.push("landlock-requested".into());
    }
    let _ = jail;
    Ok(Spawned {
        stdout,
        stderr,
        status: status.code().unwrap_or(-1),
        enforcement,
        landlock_requested: use_landlock,
    })
}

fn python_failed_to_start(spawned: &Spawned) -> bool {
    let stderr = spawned.stderr.to_ascii_lowercase();
    spawned.status != 0
        && (stderr.contains("shared librar")
            || stderr.contains("landlock")
            || stderr.contains("permission denied")
            || (spawned.status == 127))
        && !stderr.contains("modulenotfound")
}

fn summarise(events: &[Value]) -> (bool, bool, String, String) {
    let mut canary = false;
    let mut outbound = false;
    let mut filesystem = "unknown".to_string();
    let mut network = "unknown".to_string();
    for event in events {
        let probe = event.get("probe").and_then(|v| v.as_bool()).unwrap_or(false);
        if event.get("kind").and_then(|v| v.as_str()) == Some("enforcement_probe") {
            if let Some(value) = event.get("filesystem").and_then(|v| v.as_str()) {
                filesystem = value.to_string();
            }
            if let Some(value) = event.get("network").and_then(|v| v.as_str()) {
                network = value.to_string();
            }
        }
        if probe {
            continue;
        }
        match event.get("kind").and_then(|v| v.as_str()) {
            Some("canary_read") => canary = true,
            Some("outbound") => outbound = true,
            _ => {}
        }
    }
    (canary, outbound, filesystem, network)
}

fn write_canary(path: &Path, body: &str) -> Result<PathBuf, String> {
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(body.as_bytes()).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    Ok(path.to_path_buf())
}

fn tail(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        text[text.len() - max..].to_string()
    }
}

fn run_docker(args: &Args) -> Result<String, String> {
    // Optional fallback for hosts without seccomp/landlock. Requires a local
    // python image; this path is not used by the Linux acceptance test.
    let stamp = std::process::id();
    let jail = env::temp_dir().join(format!("modelsentinel-docker-{stamp}"));
    fs::create_dir_all(jail.join("home")).map_err(|e| e.to_string())?;
    fs::create_dir_all(jail.join("work")).map_err(|e| e.to_string())?;
    fs::copy(&args.code, jail.join("work/snippet.py")).map_err(|e| e.to_string())?;
    fs::write(jail.join("work/runner.py"), RUNNER).map_err(|e| e.to_string())?;
    let token = format!("docker{stamp:x}");
    fs::write(
        jail.join("home/.env"),
        format!("SENTINEL_CANARY_DOTENV={token}\n"),
    )
    .map_err(|e| e.to_string())?;
    let mount = format!("{}:/work", jail.display());
    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "-v",
            &mount,
            "-e",
            "HOME=/work/home",
            "-e",
            "SENTINEL_CODE=/work/work/snippet.py",
            "-e",
            "SENTINEL_EVENTS=/work/work/events.jsonl",
            "-e",
            &format!("SENTINEL_CANARIES=/work/home/.env"),
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "python:3.11-slim",
            "python",
            "/work/work/runner.py",
        ])
        .output()
        .map_err(|e| format!("docker: {e}"))?;
    let events_raw = fs::read_to_string(jail.join("work/events.jsonl")).unwrap_or_default();
    let events: Vec<Value> = events_raw
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let (canary_read, outbound_attempt, filesystem, network) = summarise(&events);
    let _ = fs::remove_dir_all(&jail);
    Ok(json!({
        "enforcement": ["docker", "network-none", "audit-hook"],
        "exit_code": output.status.code().unwrap_or(-1),
        "timed_out": false,
        "canary_read": canary_read,
        "outbound_attempt": outbound_attempt,
        "network_probe": network,
        "filesystem_probe": filesystem,
        "events": events,
        "stdout": tail(&String::from_utf8_lossy(&output.stdout), 2000),
        "stderr": tail(&String::from_utf8_lossy(&output.stderr), 2000),
        "canary_token": token,
        "error": null,
    })
    .to_string())
}

fn emit_error(message: &str) {
    println!(
        "{}",
        json!({
            "enforcement": [],
            "exit_code": null,
            "timed_out": false,
            "canary_read": false,
            "outbound_attempt": false,
            "events": [],
            "error": message,
        })
    );
}
