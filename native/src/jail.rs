//! Platform-neutral parts of the sandbox: the jail directory with canary
//! secrets, the child environment, and the summary of the runner's events.
//!
//! Everything here compiles on Linux, macOS and Windows. The kernel
//! restrictions live in `restrict.rs` (Linux only) and the spawning logic in
//! `spawn.rs`.

use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// The runner script, embedded so the binary is self-contained.
pub const RUNNER: &str = include_str!("sandbox_runner.py");

/// A per-run temporary directory: `home/` holds canaries, `work/` the code.
#[derive(Debug)]
pub struct Jail {
    pub root: PathBuf,
    pub home: PathBuf,
    pub work: PathBuf,
    pub snippet: PathBuf,
    pub runner: PathBuf,
    pub events: PathBuf,
    pub outside: PathBuf,
    pub token: String,
    pub canaries: Vec<PathBuf>,
}

fn stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Relative canary paths and their contents (with `{t}` replaced by the token).
const CANARY_FILES: &[(&str, &str)] = &[
    (".ssh/id_rsa", "SENTINEL_CANARY_SSH {t}\n"),
    (".ssh/authorized_keys", "SENTINEL_CANARY_AUTHKEYS {t}\n"),
    (
        ".aws/credentials",
        "[default]\naws_access_key_id = SENTINEL_CANARY_AWS_{t}\naws_secret_access_key = SENTINEL_CANARY_SECRET_{t}\n",
    ),
    (".env", "SENTINEL_CANARY_DOTENV={t}\nAPI_TOKEN=SENTINEL_CANARY_TOKEN_{t}\n"),
    (".netrc", "machine sentinel.invalid login canary password SENTINEL_CANARY_NETRC_{t}\n"),
];

impl Jail {
    /// Create the jail under `base` (normally the system temp dir).
    pub fn create(base: &Path, code: &Path) -> io::Result<Jail> {
        let now = stamp();
        let root = base.join(format!("modelsentinel-jail-{now:x}-{}", std::process::id()));
        let home = root.join("home");
        let work = root.join("work");
        fs::create_dir_all(home.join(".ssh"))?;
        fs::create_dir_all(home.join(".aws"))?;
        fs::create_dir_all(&work)?;
        let token = format!("{now:x}");
        let mut canaries = Vec::new();
        for (rel, body) in CANARY_FILES {
            let path = home.join(rel);
            write_private(&path, &body.replace("{t}", &token))?;
            canaries.push(path);
        }
        let snippet = work.join("snippet.py");
        fs::copy(code, &snippet)?;
        let runner = work.join("runner.py");
        fs::write(&runner, RUNNER)?;
        let events = work.join("events.jsonl");
        // A marker *outside* the jail. Reading it tells us whether filesystem
        // isolation is real. It sits next to the jail, not inside it.
        let outside = base.join(format!("modelsentinel-outside-{now:x}.txt"));
        fs::write(&outside, b"outside-jail-marker\n")?;
        Ok(Jail {
            root,
            home,
            work,
            snippet,
            runner,
            events,
            outside,
            token,
            canaries,
        })
    }

    /// Canary paths joined with the platform path separator (`:` or `;`).
    pub fn canary_env(&self) -> OsString {
        env::join_paths(&self.canaries).unwrap_or_default()
    }

    pub fn read_events(&self) -> Vec<Value> {
        parse_events(&fs::read_to_string(&self.events).unwrap_or_default())
    }

    /// Remove the jail unless `SENTINEL_KEEP_JAIL=1`.
    pub fn cleanup(&self) {
        let _ = fs::remove_file(&self.outside);
        if env::var("SENTINEL_KEEP_JAIL").ok().as_deref() != Some("1") {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn write_private(path: &Path, body: &str) -> io::Result<()> {
    fs::write(path, body.as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// The real user's home, so the runner can refuse reads of it.
pub fn real_home() -> Option<OsString> {
    if cfg!(windows) {
        env::var_os("USERPROFILE").or_else(|| env::var_os("HOME"))
    } else {
        env::var_os("HOME")
    }
}

/// Environment for the child. The parent calls `env_clear()` first, so this is
/// the *whole* environment. `python_dir` is added to `PATH` on Windows, where
/// the interpreter needs its own directory to find its DLLs.
pub fn child_env(jail: &Jail, python_dir: Option<&Path>) -> Vec<(OsString, OsString)> {
    let mut vars: Vec<(OsString, OsString)> = Vec::new();
    let mut set = |k: &str, v: OsString| vars.push((OsString::from(k), v));
    set("HOME", jail.home.clone().into_os_string());
    set("PYTHONUNBUFFERED", "1".into());
    set("PYTHONDONTWRITEBYTECODE", "1".into());
    set("PYTHONNOUSERSITE", "1".into());
    set("PYTHONUTF8", "1".into());
    set("PYTHONIOENCODING", "utf-8".into());
    set("SENTINEL_CODE", jail.snippet.clone().into_os_string());
    set("SENTINEL_EVENTS", jail.events.clone().into_os_string());
    set("SENTINEL_CANARIES", jail.canary_env());
    set("SENTINEL_OUTSIDE", jail.outside.clone().into_os_string());
    set("SENTINEL_TOKEN", jail.token.clone().into());
    set("SENTINEL_JAIL", jail.root.clone().into_os_string());
    set("SENTINEL_DENY", "1".into());
    if let Some(home) = real_home() {
        set("SENTINEL_REAL_HOME", home);
    }
    if cfg!(windows) {
        // Python's expanduser() reads USERPROFILE on Windows, not HOME.
        set("USERPROFILE", jail.home.clone().into_os_string());
        set("APPDATA", jail.home.join("AppData").join("Roaming").into_os_string());
        set("LOCALAPPDATA", jail.home.join("AppData").join("Local").into_os_string());
        set("TEMP", jail.work.clone().into_os_string());
        set("TMP", jail.work.clone().into_os_string());
        let system_root = env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        set("SystemRoot", system_root.clone());
        set("windir", system_root.clone());
        let mut path_dirs: Vec<PathBuf> = Vec::new();
        if let Some(dir) = python_dir {
            path_dirs.push(dir.to_path_buf());
        }
        path_dirs.push(PathBuf::from(&system_root).join("System32"));
        set("PATH", env::join_paths(path_dirs).unwrap_or_default());
        set("PATHEXT", ".COM;.EXE;.BAT;.CMD".into());
    } else {
        set("TMPDIR", jail.work.clone().into_os_string());
        set("PATH", "/usr/bin:/bin".into());
        if cfg!(target_os = "linux") {
            set("LANG", "C.UTF-8".into());
            set("LC_ALL", "C.UTF-8".into());
        }
    }
    vars
}

pub fn parse_events(raw: &str) -> Vec<Value> {
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<Value>(line)
                .unwrap_or_else(|_| json!({"kind": "parse_error", "detail": tail(line, 300)}))
        })
        .collect()
}

#[derive(Debug, Default, PartialEq)]
pub struct Summary {
    pub canary_read: bool,
    pub outbound_attempt: bool,
    pub exfil_canary: bool,
    pub process_spawn: bool,
    pub outbound_hosts: Vec<String>,
    pub dns_queries: Vec<String>,
    pub blocked_actions: usize,
    pub files_opened: usize,
    pub network_probe: String,
    pub filesystem_probe: String,
}

fn str_field<'a>(event: &'a Value, key: &str) -> Option<&'a str> {
    event.get(key).and_then(|v| v.as_str())
}

/// Reduce the runner's event list to the facts the scorer needs. Probe events
/// (the runner's own self-test) never count as snippet behaviour.
pub fn summarise(events: &[Value]) -> Summary {
    let mut s = Summary {
        network_probe: "unknown".into(),
        filesystem_probe: "unknown".into(),
        ..Summary::default()
    };
    let mut hosts = BTreeSet::new();
    let mut dns = BTreeSet::new();
    for event in events {
        let kind = str_field(event, "kind").unwrap_or("");
        if kind == "enforcement_probe" {
            if let Some(v) = str_field(event, "network") {
                s.network_probe = v.to_string();
            }
            if let Some(v) = str_field(event, "filesystem") {
                s.filesystem_probe = v.to_string();
            }
            continue;
        }
        if event.get("probe").and_then(|v| v.as_bool()).unwrap_or(false) {
            continue;
        }
        let host = str_field(event, "host").filter(|h| !h.is_empty());
        match kind {
            "canary_read" => s.canary_read = true,
            "file_access" => s.files_opened += 1,
            "dns_query" => {
                s.outbound_attempt = true;
                if event.get("canary").and_then(|v| v.as_bool()).unwrap_or(false) {
                    s.exfil_canary = true;
                }
                if let Some(h) = host {
                    dns.insert(h.to_string());
                    hosts.insert(h.to_string());
                }
            }
            "outbound" | "raw_socket" => {
                s.outbound_attempt = true;
                if let Some(h) = host {
                    hosts.insert(h.to_string());
                }
            }
            "outbound_payload" => {
                s.outbound_attempt = true;
                if let Some(h) = host {
                    hosts.insert(h.to_string());
                }
                if event.get("canary").and_then(|v| v.as_bool()).unwrap_or(false) {
                    s.exfil_canary = true;
                }
                if let Some(q) = str_field(event, "dns_qname") {
                    dns.insert(q.to_string());
                }
            }
            "process_spawn" => s.process_spawn = true,
            "blocked" => {
                s.blocked_actions += 1;
                match str_field(event, "category") {
                    Some("network") => s.outbound_attempt = true,
                    Some("process") => s.process_spawn = true,
                    _ => {}
                }
            }
            _ => {}
        }
        if kind == "process_spawn" && event.get("blocked").and_then(|v| v.as_bool()) == Some(true) {
            s.blocked_actions += 1;
        }
    }
    s.outbound_hosts = hosts.into_iter().collect();
    s.dns_queries = dns.into_iter().collect();
    s
}

/// `full` only when the probes prove the kernel denied the network *and* the
/// read outside the jail. `container` for Docker. Anything else is `reduced`.
pub fn isolation_level(enforcement: &[String], summary: &Summary) -> &'static str {
    if enforcement.iter().any(|e| e == "docker") {
        return "container";
    }
    let net = summary.network_probe == "eperm";
    let fs = summary.filesystem_probe.ends_with("-blocked");
    if net && fs {
        "full"
    } else {
        "reduced"
    }
}

pub fn isolation_notes(level: &str, summary: &Summary) -> Vec<String> {
    let mut notes = Vec::new();
    if level == "full" || level == "container" {
        return notes;
    }
    if summary.network_probe != "eperm" {
        notes.push(format!(
            "kernel network denial not in force (probe: {}); network is contained only by the in-process sink and audit-hook denial",
            summary.network_probe
        ));
    }
    if !summary.filesystem_probe.ends_with("-blocked") {
        notes.push(format!(
            "filesystem is not kernel-restricted (probe: {}); reads outside the jail are limited only by the audit hook",
            summary.filesystem_probe
        ));
    }
    notes
}

/// Keep at most `max` bytes from the end, cut on a UTF-8 boundary.
pub fn tail(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("sentinel-jail-test-{name}-{}", stamp()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn jail_writes_canaries_with_token_and_cleans_up() {
        let base = scratch("create");
        let code = base.join("code.py");
        fs::write(&code, "print(1)\n").unwrap();
        let jail = Jail::create(&base, &code).unwrap();
        assert_eq!(jail.canaries.len(), CANARY_FILES.len());
        let aws = fs::read_to_string(jail.home.join(".aws").join("credentials")).unwrap();
        assert!(aws.contains(&jail.token));
        assert!(jail.runner.is_file() && jail.snippet.is_file() && jail.outside.is_file());
        let joined = jail.canary_env();
        let split: Vec<PathBuf> = env::split_paths(&joined).collect();
        assert_eq!(split, jail.canaries);
        jail.cleanup();
        assert!(!jail.root.exists());
        assert!(!jail.outside.exists());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn child_env_points_home_at_jail_and_has_no_parent_leak() {
        let base = scratch("env");
        let code = base.join("code.py");
        fs::write(&code, "").unwrap();
        let jail = Jail::create(&base, &code).unwrap();
        let vars = child_env(&jail, None);
        let get = |k: &str| vars.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(get("HOME"), Some(jail.home.clone().into_os_string()));
        if cfg!(windows) {
            assert_eq!(get("USERPROFILE"), Some(jail.home.clone().into_os_string()));
            assert!(get("SystemRoot").is_some());
        } else {
            assert_eq!(get("PATH"), Some(OsString::from("/usr/bin:/bin")));
        }
        assert!(get("AWS_ACCESS_KEY_ID").is_none());
        jail.cleanup();
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn summarise_collects_hosts_payloads_and_ignores_probe() {
        let events = parse_events(
            r#"{"kind":"enforcement_probe","probe":true,"network":"eperm","filesystem":"landlock-blocked"}
{"kind":"outbound","probe":true,"host":"127.0.0.1"}
{"kind":"canary_read","path":"/j/home/.aws/credentials"}
{"kind":"dns_query","host":"exfil.example"}
{"kind":"outbound","host":"exfil.example","port":80}
{"kind":"outbound_payload","host":"exfil.example","port":80,"canary":true}
{"kind":"outbound_payload","host":"8.8.8.8","port":53,"dns_qname":"abc.tunnel.example"}
{"kind":"file_access","path":"/j/work/out.txt"}
{"kind":"blocked","category":"process","event":"subprocess.Popen"}
not json"#,
        );
        let s = summarise(&events);
        assert!(s.canary_read && s.outbound_attempt && s.exfil_canary && s.process_spawn);
        assert_eq!(s.outbound_hosts, vec!["8.8.8.8", "exfil.example"]);
        assert_eq!(s.dns_queries, vec!["abc.tunnel.example", "exfil.example"]);
        assert_eq!(s.blocked_actions, 1);
        assert_eq!(s.files_opened, 1);
        assert_eq!(isolation_level(&[], &s), "full");
        assert_eq!(events.last().unwrap()["kind"], "parse_error");
    }

    #[test]
    fn isolation_is_reduced_without_kernel_probes() {
        let s = Summary {
            network_probe: "connected".into(),
            filesystem_probe: "outside-readable".into(),
            ..Summary::default()
        };
        assert_eq!(isolation_level(&["audit-hook".into()], &s), "reduced");
        assert_eq!(isolation_notes("reduced", &s).len(), 2);
        assert_eq!(isolation_level(&["docker".into()], &s), "container");
    }

    #[test]
    fn tail_respects_utf8_boundaries() {
        let text = "aé".repeat(10);
        let cut = tail(&text, 5);
        assert!(cut.len() <= 5);
        assert!(text.ends_with(&cut));
    }
}
