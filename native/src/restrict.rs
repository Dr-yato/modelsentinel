//! seccomp network deny-list and Landlock filesystem policy.
//!
//! These are applied in the forked child before `exec`. They are a containment
//! layer for generated code, not a claim that a hostile program cannot escape.

use std::io;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SockFilter {
    pub code: u16,
    pub jt: u8,
    pub jf: u8,
    pub k: u32,
}

#[repr(C)]
struct SockFprog {
    len: u16,
    filter: *const SockFilter,
}

const BPF_LD_W_ABS: u16 = 0x20;
const BPF_JMP_JEQ_K: u16 = 0x15;
const BPF_RET_K: u16 = 0x06;
const SECCOMP_RET_KILL: u32 = 0x8000_0000;
const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
const SECCOMP_SET_MODE_FILTER: i64 = 1;
const EPERM: u32 = 1;

fn stmt(code: u16, k: u32) -> SockFilter {
    SockFilter {
        code,
        jt: 0,
        jf: 0,
        k,
    }
}

fn jeq(k: u32, jt: u8, jf: u8) -> SockFilter {
    SockFilter {
        code: BPF_JMP_JEQ_K,
        jt,
        jf,
        k,
    }
}

fn audit_arch() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        0xc000_003e
    }
    #[cfg(target_arch = "aarch64")]
    {
        0xc000_00b7
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        0
    }
}

fn denied_syscalls() -> &'static [u32] {
    #[cfg(target_arch = "x86_64")]
    {
        &[
            41,  // socket
            42,  // connect
            43,  // accept
            44,  // sendto
            45,  // recvfrom
            46,  // sendmsg
            47,  // recvmsg
            48,  // shutdown
            49,  // bind
            50,  // listen
            51,  // getsockname
            52,  // getpeername
            53,  // socketpair
            101, // ptrace
            288, // accept4
            299, // recvmmsg
            307, // sendmmsg
            425, // io_uring_setup
            426, // io_uring_enter
            427, // io_uring_register
        ]
    }
    #[cfg(target_arch = "aarch64")]
    {
        &[
            198, // socket
            199, // socketpair
            200, // bind
            201, // listen
            202, // accept
            203, // connect
            204, // getsockname
            205, // getpeername
            206, // sendto
            207, // recvfrom
            210, // shutdown
            211, // sendmsg
            212, // recvmsg
            242, // accept4
            117, // ptrace
        ]
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        &[]
    }
}

pub fn network_filter() -> Vec<SockFilter> {
    let mut prog = Vec::new();
    prog.push(stmt(BPF_LD_W_ABS, 4));
    prog.push(jeq(audit_arch(), 1, 0));
    prog.push(stmt(BPF_RET_K, SECCOMP_RET_KILL));
    prog.push(stmt(BPF_LD_W_ABS, 0));
    for nr in denied_syscalls() {
        prog.push(jeq(*nr, 0, 1));
        prog.push(stmt(BPF_RET_K, SECCOMP_RET_ERRNO | EPERM));
    }
    prog.push(stmt(BPF_RET_K, SECCOMP_RET_ALLOW));
    prog
}

pub fn apply_seccomp(filter: &[SockFilter]) -> io::Result<()> {
    if filter.is_empty() || audit_arch() == 0 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "seccomp is not supported on this architecture",
        ));
    }
    let prog = SockFprog {
        len: filter.len() as u16,
        filter: filter.as_ptr(),
    };
    let pr = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if pr != 0 {
        return Err(io::Error::last_os_error());
    }
    let rc = unsafe { libc::syscall(libc::SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

const LANDLOCK_CREATE: i64 = 444;
const LANDLOCK_ADD_RULE: i64 = 445;
const LANDLOCK_RESTRICT: i64 = 446;
const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;

const FS_EXECUTE: u64 = 1 << 0;
const FS_WRITE: u64 = 1 << 1;
const FS_READ: u64 = 1 << 2;
const FS_READDIR: u64 = 1 << 3;
const FS_REMOVE_DIR: u64 = 1 << 4;
const FS_REMOVE_FILE: u64 = 1 << 5;
const FS_MAKE_DIR: u64 = 1 << 7;
const FS_MAKE_REG: u64 = 1 << 8;

#[repr(C)]
struct RulesetAttr {
    handled_access_fs: u64,
}

#[repr(C, packed)]
struct PathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

fn landlock_abi() -> i32 {
    let rc = unsafe { libc::syscall(LANDLOCK_CREATE, std::ptr::null::<u8>(), 0usize, 1u32) };
    if rc < 0 {
        0
    } else {
        rc as i32
    }
}

fn handled_access(abi: i32) -> u64 {
    let mut mask = (1u64 << 13) - 1; // ABI 1 bits 0..=12
    if abi >= 2 {
        mask |= 1 << 13;
    }
    if abi >= 3 {
        mask |= 1 << 14;
    }
    if abi >= 4 {
        mask |= 1 << 15;
    }
    mask
}

pub struct LandlockFds {
    pub read: Vec<i32>,
    pub write: Vec<i32>,
}

pub fn open_path(path: &str) -> Option<i32> {
    let c = std::ffi::CString::new(path).ok()?;
    let flags = libc::O_PATH | libc::O_CLOEXEC;
    let fd = unsafe { libc::open(c.as_ptr(), flags) };
    if fd < 0 {
        None
    } else {
        Some(fd)
    }
}

pub fn apply_landlock(fds: &LandlockFds) -> io::Result<()> {
    let abi = landlock_abi();
    if abi < 1 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "landlock is not available",
        ));
    }
    let attr = RulesetAttr {
        handled_access_fs: handled_access(abi),
    };
    let ruleset = unsafe {
        libc::syscall(
            LANDLOCK_CREATE,
            &attr as *const RulesetAttr,
            std::mem::size_of::<RulesetAttr>(),
            0u32,
        )
    };
    if ruleset < 0 {
        return Err(io::Error::last_os_error());
    }
    let read_access = FS_EXECUTE | FS_READ | FS_READDIR;
    let write_access = read_access
        | FS_WRITE
        | FS_REMOVE_DIR
        | FS_REMOVE_FILE
        | FS_MAKE_DIR
        | FS_MAKE_REG;
    for fd in &fds.read {
        add_rule(ruleset, *fd, read_access)?;
    }
    for fd in &fds.write {
        add_rule(ruleset, *fd, write_access)?;
    }
    let rc = unsafe { libc::syscall(LANDLOCK_RESTRICT, ruleset, 0u32) };
    unsafe { libc::close(ruleset as i32) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn add_rule(ruleset: i64, fd: i32, access: u64) -> io::Result<()> {
    let beneath = PathBeneathAttr {
        allowed_access: access,
        parent_fd: fd,
    };
    let rc = unsafe {
        libc::syscall(
            LANDLOCK_ADD_RULE,
            ruleset,
            LANDLOCK_RULE_PATH_BENEATH,
            &beneath as *const PathBeneathAttr,
            0u32,
        )
    };
    if rc != 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn close_fds(fds: &LandlockFds) {
    for fd in fds.read.iter().chain(fds.write.iter()) {
        unsafe { libc::close(*fd) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seccomp_denies_socket_in_child() {
        let filter = network_filter();
        let filter: &'static [SockFilter] = Box::leak(filter.into_boxed_slice());
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            if apply_seccomp(filter).is_err() {
                unsafe { libc::_exit(2) };
            }
            let rc = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
            let err = std::io::Error::last_os_error();
            if rc < 0 && err.raw_os_error() == Some(1) {
                unsafe { libc::_exit(0) };
            }
            unsafe { libc::_exit(1) };
        }
        let mut status = 0;
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(waited, pid);
        let code = libc::WEXITSTATUS(status);
        assert_eq!(code, 0, "child exit {code}, raw status {status}");
    }
}
