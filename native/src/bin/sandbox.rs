//! Run one generated Python snippet in a throwaway jail and report what it did.
//!
//! Every platform gets: a cleared environment, a temp HOME/USERPROFILE holding
//! canary secrets, an audit-hook runner that logs every file open, an
//! in-process DNS/HTTP sink that records hostnames and payloads, and hook-level
//! denial of process spawn / raw sockets / writes outside the jail.
//!
//! On top of that, per platform:
//! - Linux: seccomp denies socket syscalls and Landlock confines the
//!   filesystem. When both probes confirm it, isolation is `full`.
//! - macOS: `sandbox-exec` (when present) denies network and writes outside
//!   the jail. Isolation is reported as `reduced` because reads are not
//!   confined.
//! - Windows and others: hook-level containment only, isolation `reduced`.
//! - `--docker`: `--network none` container from a *local* image (never
//!   pulled), isolation `container`.
//!
//! Only Python is executed. Other languages are refused, not run.

use sentinel_native::jail::{
    child_env, isolation_level, isolation_notes, summarise, tail, Jail,
};
use sentinel_native::spawn::{find_program, resolve_python, run_with_timeout, Outcome};
use serde_json::{json, Value};
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

const USAGE: &str = "usage: sentinel-sandbox --code FILE [--timeout SECONDS] [--lang python] [--docker] [--docker-image IMAGE] | --probe";
const DEFAULT_IMAGE: &str = "python:3.11-slim";
const MAX_EVENTS_OUT: usize = 1500;

struct Args {
    code: PathBuf,
    timeout: Duration,
    docker: bool,
    image: String,
    lang: String,
}

/// What a platform backend hands back to the common reporting code.
struct Run {
    outcome: Outcome,
    enforcement: Vec<String>,
    restrict_status: String,
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            emit_error(&message, "none");
            std::process::exit(2);
        }
    };
    if !matches!(args.lang.as_str(), "python" | "py") {
        emit_error(
            &format!(
                "unsupported language '{}': only Python snippets are executed; JavaScript and shell are analysed statically but never run",
                args.lang
            ),
            "none",
        );
        std::process::exit(3);
    }
    let result = if args.docker { run_docker(&args) } else { run_native(&args) };
    match result {
        Ok(value) => println!("{value}"),
        Err(message) => {
            emit_error(&message, "none");
            std::process::exit(1);
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut code = None;
    let mut timeout = 5u64;
    let mut docker = false;
    let mut image = env::var("SENTINEL_DOCKER_IMAGE").unwrap_or_else(|_| DEFAULT_IMAGE.into());
    let mut lang = "python".to_string();
    let mut iter = env::args_os().skip(1);
    while let Some(arg) = iter.next() {
        let mut value = |name: &str| -> Result<OsString, String> {
            iter.next().ok_or_else(|| format!("missing {name} value"))
        };
        match arg.to_str().unwrap_or("") {
            "--code" => code = Some(PathBuf::from(value("--code")?)),
            "--timeout" => {
                let raw = value("--timeout")?;
                timeout = raw
                    .to_str()
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or("timeout must be whole seconds")?;
            }
            "--lang" => lang = value("--lang")?.to_string_lossy().to_ascii_lowercase(),
            "--docker" => docker = true,
            "--probe" => {
                println!("{}", kernel_probe());
                std::process::exit(0);
            }
            "--docker-image" => image = value("--docker-image")?.to_string_lossy().into_owned(),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let code = code.ok_or(USAGE)?;
    if !code.is_file() {
        return Err(format!("code file not found: {}", code.display()));
    }
    Ok(Args {
        code,
        timeout: Duration::from_secs(timeout.clamp(1, 600)),
        docker,
        image,
        lang,
    })
}

/// What this kernel/build can enforce, without running anything.
fn kernel_probe() -> Value {
    use sentinel_native::restrict::{handled_access, kernel_isolation_supported, landlock_abi_version};
    let abi = landlock_abi_version();
    json!({
        "os": env::consts::OS,
        "seccomp_supported": kernel_isolation_supported(),
        "landlock_abi": abi,
        "landlock_handled_access": handled_access(abi),
    })
}

fn base_enforcement() -> Vec<String> {
    ["audit-hook", "cleared-env", "canary-secrets", "in-process-sink", "hook-deny"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn base_command(program: &Path, args: &[OsString], jail: &Jail, envs: &[(OsString, OsString)]) -> Command {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .env_clear()
        .envs(envs.iter().map(|(k, v)| (k, v)))
        .current_dir(&jail.work);
    cmd
}

#[cfg_attr(not(any(target_os = "linux", target_os = "macos")), allow(dead_code))]
/// The runner never reached its probe: the interpreter (or the restriction
/// around it) failed before any Python ran.
fn runner_never_started(jail: &Jail, outcome: &Outcome) -> bool {
    !outcome.timed_out
        && !jail
            .read_events()
            .iter()
            .any(|e| e.get("kind").and_then(|v| v.as_str()) == Some("enforcement_probe"))
}

fn run_native(args: &Args) -> Result<String, String> {
    let python = resolve_python()?;
    let jail = Jail::create(&env::temp_dir(), &args.code).map_err(|e| format!("jail: {e}"))?;
    let envs = child_env(&jail, python.parent());
    let run = platform::run(&python, &jail, &envs, args.timeout);
    let value = run.map(|run| report(&jail, run, "native"));
    jail.cleanup();
    value.map(|v| v.to_string())
}

fn report(jail: &Jail, run: Run, backend: &str) -> Value {
    let events = jail.read_events();
    let summary = summarise(&events);
    let mut enforcement = run.enforcement;
    if cfg!(target_os = "linux") && backend == "native" {
        if summary.network_probe == "eperm" {
            enforcement.push("seccomp".into());
        }
        if summary.filesystem_probe == "landlock-blocked" {
            enforcement.push("landlock".into());
        }
    }
    if cfg!(target_os = "macos") && summary.network_probe == "eperm" && enforcement.iter().any(|e| e == "sandbox-exec") {
        enforcement.push("sandbox-exec-network".into());
    }
    let isolation = isolation_level(&enforcement, &summary);
    let notes = isolation_notes(isolation, &summary);
    let truncated = events.len() > MAX_EVENTS_OUT;
    let shown: Vec<Value> = events.into_iter().take(MAX_EVENTS_OUT).collect();
    json!({
        "backend": backend,
        "platform": env::consts::OS,
        "language": "python",
        "executed": true,
        "isolation": isolation,
        "isolation_notes": notes,
        "enforcement": enforcement,
        "exit_code": run.outcome.status,
        "timed_out": run.outcome.timed_out,
        "canary_read": summary.canary_read,
        "outbound_attempt": summary.outbound_attempt,
        "outbound_hosts": summary.outbound_hosts,
        "dns_queries": summary.dns_queries,
        "exfil_canary": summary.exfil_canary,
        "process_spawn": summary.process_spawn,
        "blocked_actions": summary.blocked_actions,
        "files_opened": summary.files_opened,
        "network_probe": summary.network_probe,
        "filesystem_probe": summary.filesystem_probe,
        "events": shown,
        "events_truncated": truncated,
        "stdout": tail(&run.outcome.stdout, 2000),
        "stderr": tail(&run.outcome.stderr, 2000),
        "canary_token": jail.token,
        "jail": jail.root.display().to_string(),
        "restrict_status": run.restrict_status,
        "error": Value::Null,
    })
}

// ---------------------------------------------------------------------------
// Linux: seccomp + Landlock applied in the child between fork and exec.
// ---------------------------------------------------------------------------
#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    #[allow(unused_imports)]
    use std::fs;
    use sentinel_native::restrict::{
        apply_landlock, apply_seccomp, close_fds, last_errno, network_filter, open_path, set_no_new_privs,
        LandlockFds,
    };
    use std::os::fd::IntoRawFd;
    use std::os::unix::process::CommandExt;

    const READ_ROOTS: &[&str] = &["/usr", "/lib", "/lib64", "/etc", "/dev", "/proc", "/bin", "/sys", "/opt"];

    fn landlock_fds(jail: &Jail, python: &Path) -> LandlockFds {
        let mut read: Vec<i32> = READ_ROOTS
            .iter()
            .filter(|p| Path::new(p).exists())
            .filter_map(|p| open_path(p))
            .collect();
        // An interpreter outside the standard roots (pyenv, uv, a venv in
        // $HOME) needs its prefixes readable. Both are needed for a venv:
        // `.venv/bin/python` is a symlink, so the canonical prefix is the base
        // install (stdlib), while Python reads `.venv/pyvenv.cfg` at startup
        // from the *non-canonical* prefix. Without it, Python dies with EACCES
        // under Landlock and the run silently fell back to no Landlock.
        for prefix in interpreter_prefixes(python) {
            if let Some(fd) = open_path(&prefix.display().to_string()) {
                read.push(fd);
            }
        }
        let write = open_path(&jail.root.display().to_string()).into_iter().collect();
        LandlockFds { read, write }
    }

    /// `<prefix>` of `<prefix>/bin/python`, for both the path as given
    /// (a venv) and its canonical target (the base interpreter).
    pub(super) fn interpreter_prefixes(python: &Path) -> Vec<PathBuf> {
        let mut prefixes: Vec<PathBuf> = Vec::new();
        let absolute = if python.is_absolute() {
            python.to_path_buf()
        } else {
            env::current_dir().map(|cwd| cwd.join(python)).unwrap_or_else(|_| python.to_path_buf())
        };
        let canonical = fs::canonicalize(python).ok();
        for path in [Some(absolute), canonical].into_iter().flatten() {
            if let Some(prefix) = path.parent().and_then(|bin| bin.parent()) {
                if prefix != Path::new("/") && !prefixes.iter().any(|p| p == prefix) {
                    prefixes.push(prefix.to_path_buf());
                }
            }
        }
        prefixes
    }

    fn spawn(python: &Path, jail: &Jail, envs: &[(OsString, OsString)], timeout: Duration, landlock: bool) -> Result<Run, String> {
        let fds = landlock_fds(jail, python);
        let use_landlock = landlock && !fds.write.is_empty();
        let status_path = jail.work.join("restrict.txt");
        let status_fd = fs::File::create(&status_path).map_err(|e| e.to_string())?.into_raw_fd();
        let filter = network_filter();
        let mut cmd = base_command(python, &[jail.runner.clone().into_os_string()], jail, envs);
        let child_fds = LandlockFds { read: fds.read.clone(), write: fds.write.clone() };
        // SAFETY: the closure runs in the forked child before exec. It only
        // calls prctl/landlock/seccomp syscalls and write(2) on an inherited
        // fd, all async-signal-safe, and allocates nothing.
        unsafe {
            cmd.pre_exec(move || {
                set_no_new_privs();
                if !use_landlock {
                    let msg = b"landlock-skipped\n";
                    libc::write(status_fd, msg.as_ptr() as *const libc::c_void, msg.len());
                } else if apply_landlock(&child_fds).is_ok() {
                    let msg = b"landlock-ok\n";
                    libc::write(status_fd, msg.as_ptr() as *const libc::c_void, msg.len());
                } else {
                    // Record why (errno) without allocating: EINVAL means the
                    // ruleset named a right this kernel's ABI does not know.
                    let errno = last_errno();
                    let mut buf = *b"landlock-err:000\n";
                    let mut value = errno.clamp(0, 999) as u32;
                    for slot in (13..16).rev() {
                        buf[slot] = b'0' + (value % 10) as u8;
                        value /= 10;
                    }
                    libc::write(status_fd, buf.as_ptr() as *const libc::c_void, buf.len());
                }
                let msg: &[u8] = if apply_seccomp(&filter).is_ok() { b"seccomp-ok\n" } else { b"seccomp-err\n" };
                libc::write(status_fd, msg.as_ptr() as *const libc::c_void, msg.len());
                Ok(())
            });
        }
        let outcome = run_with_timeout(cmd, timeout).map_err(|e| format!("spawn python: {e}"));
        // SAFETY: the parent's copy of the status fd is no longer needed.
        unsafe { libc::close(status_fd) };
        close_fds(&fds);
        let outcome = outcome?;
        let mut enforcement = base_enforcement();
        enforcement.push("seccomp-requested".into());
        if use_landlock {
            enforcement.push("landlock-requested".into());
        }
        let restrict_status = fs::read_to_string(&status_path).unwrap_or_default().trim().replace('\n', ",");
        Ok(Run { outcome, enforcement, restrict_status })
    }

    pub fn run(python: &Path, jail: &Jail, envs: &[(OsString, OsString)], timeout: Duration) -> Result<Run, String> {
        let first = spawn(python, jail, envs, timeout, true)?;
        if !runner_never_started(jail, &first.outcome) {
            return Ok(first);
        }
        // Landlock can stop an interpreter that lives outside the read roots.
        // Retry with seccomp only, and say so.
        let mut second = spawn(python, jail, envs, timeout, false)?;
        second.restrict_status = format!("{};retried-without-landlock", second.restrict_status);
        Ok(second)
    }
}

// ---------------------------------------------------------------------------
// macOS: sandbox-exec profile denying network and writes outside the jail.
// ---------------------------------------------------------------------------
#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    #[allow(unused_imports)]
    use std::fs;

    const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
    const SECRET_DIRS: &[&str] = &[".ssh", ".aws", ".gnupg", ".config", ".kube", ".docker", ".netrc", ".env", "Library/Keychains"];

    fn quote(path: &Path) -> String {
        path.display().to_string().replace('\\', "\\\\").replace('"', "\\\"")
    }

    fn profile(jail: &Jail) -> String {
        let root = fs::canonicalize(&jail.root).unwrap_or_else(|_| jail.root.clone());
        let mut p = String::from("(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n");
        p.push_str(&format!(
            "(allow file-write* (subpath \"{}\") (literal \"/dev/null\") (literal \"/dev/tty\") (regex #\"^/dev/fd/\"))\n",
            quote(&root)
        ));
        if let Some(home) = sentinel_native::jail::real_home() {
            let home = fs::canonicalize(PathBuf::from(&home)).unwrap_or_else(|_| PathBuf::from(home));
            for dir in SECRET_DIRS {
                p.push_str(&format!("(deny file-read* (subpath \"{}\"))\n", quote(&home.join(dir))));
            }
        }
        p
    }

    fn plain(python: &Path, jail: &Jail, envs: &[(OsString, OsString)], timeout: Duration) -> Result<Run, String> {
        let cmd = base_command(python, &[jail.runner.clone().into_os_string()], jail, envs);
        let outcome = run_with_timeout(cmd, timeout).map_err(|e| format!("spawn python: {e}"))?;
        Ok(Run { outcome, enforcement: base_enforcement(), restrict_status: "sandbox-exec-unavailable".into() })
    }

    pub fn run(python: &Path, jail: &Jail, envs: &[(OsString, OsString)], timeout: Duration) -> Result<Run, String> {
        if !Path::new(SANDBOX_EXEC).is_file() || env::var_os("SENTINEL_NO_SANDBOX_EXEC").is_some() {
            return plain(python, jail, envs, timeout);
        }
        let args = [
            OsString::from("-p"),
            OsString::from(profile(jail)),
            python.as_os_str().to_os_string(),
            jail.runner.clone().into_os_string(),
        ];
        let cmd = base_command(Path::new(SANDBOX_EXEC), &args, jail, envs);
        let outcome = run_with_timeout(cmd, timeout).map_err(|e| format!("spawn sandbox-exec: {e}"))?;
        if runner_never_started(jail, &outcome) {
            let mut run = plain(python, jail, envs, timeout)?;
            run.restrict_status = format!("sandbox-exec-failed:{}", tail(outcome.stderr.trim(), 200));
            return Ok(run);
        }
        let mut enforcement = base_enforcement();
        enforcement.push("sandbox-exec".into());
        Ok(Run { outcome, enforcement, restrict_status: "sandbox-exec-ok".into() })
    }
}

// ---------------------------------------------------------------------------
// Windows and other hosts: cleared env + hook-level containment only.
// ---------------------------------------------------------------------------
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use super::*;
    #[allow(unused_imports)]
    use std::fs;

    pub fn run(python: &Path, jail: &Jail, envs: &[(OsString, OsString)], timeout: Duration) -> Result<Run, String> {
        let cmd = base_command(python, &[jail.runner.clone().into_os_string()], jail, envs);
        let outcome = run_with_timeout(cmd, timeout).map_err(|e| format!("spawn python: {e}"))?;
        Ok(Run {
            outcome,
            enforcement: base_enforcement(),
            restrict_status: "no-kernel-isolation-on-this-os".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Docker: --network none from a local image. Never pulls.
// ---------------------------------------------------------------------------
fn container_path(jail: &Jail, host: &Path) -> String {
    let rel = host.strip_prefix(&jail.root).unwrap_or(host);
    let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    format!("/sandbox/{}", parts.join("/"))
}

fn run_docker(args: &Args) -> Result<String, String> {
    let docker = find_program(if cfg!(windows) { "docker.exe" } else { "docker" })
        .ok_or("docker was not found on PATH")?;
    let inspect = Command::new(&docker)
        .args(["image", "inspect", "--format", "{{.Id}}", &args.image])
        .output()
        .map_err(|e| format!("docker: {e}"))?;
    if !inspect.status.success() {
        return Err(format!(
            "docker image {} is not available locally; sentinel never pulls images. Run `docker pull {}` yourself if you want this backend",
            args.image, args.image
        ));
    }
    let jail = Jail::create(&env::temp_dir(), &args.code).map_err(|e| format!("jail: {e}"))?;
    let name = format!("modelsentinel-{}", jail.token);
    let canaries = jail.canaries.iter().map(|p| container_path(&jail, p)).collect::<Vec<_>>().join(":");
    let mount = format!("type=bind,source={},target=/sandbox", jail.root.display());
    let envs = [
        ("HOME", "/sandbox/home".to_string()),
        ("TMPDIR", "/sandbox/work".into()),
        ("SENTINEL_CODE", container_path(&jail, &jail.snippet)),
        ("SENTINEL_EVENTS", container_path(&jail, &jail.events)),
        ("SENTINEL_CANARIES", canaries),
        ("SENTINEL_TOKEN", jail.token.clone()),
        ("SENTINEL_JAIL", "/sandbox".into()),
        ("SENTINEL_DENY", "1".into()),
        ("PYTHONDONTWRITEBYTECODE", "1".into()),
        ("PYTHONUNBUFFERED", "1".into()),
    ];
    let mut cmd = Command::new(&docker);
    cmd.args([
        "run", "--rm", "--pull", "never", "--name", &name, "--network", "none", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", "512m", "--read-only",
        "--tmpfs", "/tmp:size=16m", "--mount", &mount,
    ]);
    for (k, v) in &envs {
        cmd.arg("-e").arg(format!("{k}={v}"));
    }
    cmd.args([args.image.as_str(), "python", &container_path(&jail, &jail.runner)]);
    let outcome = run_with_timeout(cmd, args.timeout + Duration::from_secs(10));
    if matches!(&outcome, Ok(o) if o.timed_out) {
        let _ = Command::new(&docker).args(["kill", &name]).output();
    }
    let outcome = match outcome {
        Ok(o) => o,
        Err(e) => {
            jail.cleanup();
            return Err(format!("docker run: {e}"));
        }
    };
    let mut enforcement = vec!["docker".to_string(), "network-none".into(), "cap-drop-all".into(), "read-only-rootfs".into()];
    enforcement.extend(base_enforcement());
    let value = report(&jail, Run { outcome, enforcement, restrict_status: format!("image={}", args.image) }, "docker");
    jail.cleanup();
    Ok(value.to_string())
}

fn emit_error(message: &str, isolation: &str) {
    println!(
        "{}",
        json!({
            "backend": "none",
            "platform": env::consts::OS,
            "executed": false,
            "isolation": isolation,
            "isolation_notes": [],
            "enforcement": [],
            "exit_code": Value::Null,
            "timed_out": false,
            "canary_read": false,
            "outbound_attempt": false,
            "outbound_hosts": [],
            "dns_queries": [],
            "exfil_canary": false,
            "process_spawn": false,
            "blocked_actions": 0,
            "events": [],
            "error": message,
        })
    );
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::platform::interpreter_prefixes;
    use std::path::PathBuf;

    #[test]
    fn venv_and_base_prefixes_are_both_read_roots() {
        let root = std::env::temp_dir().join(format!("ms-prefix-{}", std::process::id()));
        let base_bin = root.join("base/bin");
        let venv_bin = root.join("venv/bin");
        std::fs::create_dir_all(&base_bin).unwrap();
        std::fs::create_dir_all(&venv_bin).unwrap();
        std::fs::write(base_bin.join("python3.12"), b"").unwrap();
        std::os::unix::fs::symlink(base_bin.join("python3.12"), venv_bin.join("python")).unwrap();
        let prefixes = interpreter_prefixes(&venv_bin.join("python"));
        let _ = std::fs::remove_dir_all(&root);
        let canonical_base = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("ms-prefix-{}", std::process::id())).join("base");
        assert!(prefixes.contains(&root.join("venv")), "venv prefix (pyvenv.cfg) missing: {prefixes:?}");
        assert!(prefixes.iter().any(|p| p == &root.join("base") || p == &canonical_base), "base prefix missing: {prefixes:?}");
    }

    #[test]
    fn root_is_never_a_read_root() {
        assert!(!interpreter_prefixes(&PathBuf::from("/bin/python3")).contains(&PathBuf::from("/")));
    }
}
