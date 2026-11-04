//! Work limits shared by every parser.
//!
//! Untrusted files decide how much work a naive parser does: a zip bomb picks
//! the inflated size, a pickle picks the opcode count, a GGUF picks array
//! lengths. Every parser here charges a `Budget` instead. When a limit trips,
//! the parser stops that piece of work and records why, and the caller reports
//! `scan_limit_exceeded`. Hitting a limit never hangs, panics, or aborts on OOM.
//!
//! Only `std` is used, so the same code runs on Linux, macOS and Windows.

use std::time::{Duration, Instant};

/// Largest single member (zip entry, tar entry, gzip body) held in memory.
pub const MAX_MEMBER: usize = 32 * 1024 * 1024;
/// Total bytes inflated from all compressed members of one top-level file.
pub const MAX_TOTAL_INFLATE: u64 = 512 * 1024 * 1024;
/// Archive members visited per top-level file, across all nesting levels.
pub const MAX_MEMBERS: usize = 10_000;
/// Container nesting (zip in zip in gzip ...).
pub const MAX_DEPTH: u32 = 4;
/// Pickle opcodes executed symbolically per file.
pub const MAX_PICKLE_OPS: usize = 5_000_000;
/// Entries kept in a symbolic pickle memo.
pub const MAX_MEMO: usize = 1_000_000;
/// Wall-clock limit for one top-level parse.
pub const DEFAULT_TIME_LIMIT: Duration = Duration::from_secs(60);
/// Env var that overrides the wall-clock limit (whole seconds).
pub const TIME_LIMIT_ENV: &str = "SENTINEL_PARSE_TIME_LIMIT_SECS";

#[derive(Debug)]
pub struct Budget {
    deadline: Instant,
    inflated: u64,
    members: usize,
    exceeded: Vec<String>,
}

impl Default for Budget {
    fn default() -> Self {
        Self::new()
    }
}

impl Budget {
    pub fn new() -> Self {
        let limit = std::env::var(TIME_LIMIT_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_TIME_LIMIT);
        Self::with_time_limit(limit)
    }

    pub fn with_time_limit(limit: Duration) -> Self {
        Self {
            deadline: Instant::now() + limit,
            inflated: 0,
            members: 0,
            exceeded: Vec::new(),
        }
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// False once the wall clock is spent. Records the limit once.
    pub fn time_ok(&mut self) -> bool {
        if Instant::now() < self.deadline {
            return true;
        }
        self.note("time limit reached; remaining content was not scanned");
        false
    }

    /// Reserve `n` inflated bytes. False (and recorded) when the total cap is hit.
    pub fn charge_inflate(&mut self, n: u64) -> bool {
        let next = self.inflated.saturating_add(n);
        if next > MAX_TOTAL_INFLATE {
            self.note(&format!(
                "total decompressed bytes would exceed {MAX_TOTAL_INFLATE}; possible decompression bomb"
            ));
            return false;
        }
        self.inflated = next;
        true
    }

    /// Bytes still available for inflation.
    pub fn inflate_left(&self) -> u64 {
        MAX_TOTAL_INFLATE.saturating_sub(self.inflated)
    }

    /// Count one archive member. False (and recorded) once the cap is hit.
    pub fn charge_member(&mut self) -> bool {
        self.members += 1;
        if self.members > MAX_MEMBERS {
            self.note(&format!("archive member limit {MAX_MEMBERS} reached"));
            return false;
        }
        true
    }

    /// Record a limit message once.
    pub fn note(&mut self, msg: &str) {
        if !self.exceeded.iter().any(|m| m == msg) {
            self.exceeded.push(msg.to_string());
        }
    }

    pub fn exceeded(&self) -> &[String] {
        &self.exceeded
    }

    pub fn take_exceeded(&mut self) -> Vec<String> {
        std::mem::take(&mut self.exceeded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflate_cap_trips_once() {
        let mut b = Budget::new();
        assert!(b.charge_inflate(MAX_TOTAL_INFLATE));
        assert!(!b.charge_inflate(1));
        assert!(!b.charge_inflate(1));
        assert_eq!(b.exceeded().len(), 1);
    }

    #[test]
    fn zero_time_limit_is_exceeded() {
        let mut b = Budget::with_time_limit(Duration::from_secs(0));
        assert!(!b.time_ok());
        assert!(b.exceeded()[0].contains("time limit"));
    }

    #[test]
    fn member_cap() {
        let mut b = Budget::new();
        for _ in 0..MAX_MEMBERS {
            assert!(b.charge_member());
        }
        assert!(!b.charge_member());
    }
}
