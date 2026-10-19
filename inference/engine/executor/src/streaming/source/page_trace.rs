//! Opt-in Linux source observations. No payload is read by the residency probe.
//! Probe and logging costs make this diagnostic unsuitable for speed claims.

use super::{Access, ModelFile, Priority, Range};
use std::fs::File;
use std::os::{fd::AsRawFd, unix::fs::MetadataExt};
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static NEXT_SOURCE: AtomicU64 = AtomicU64::new(1);

pub(super) struct Trace {
    pub(super) id: u64,
    epoch: Instant,
    unix_ns: u128,
    owned: Vec<bool>,
    next_read: AtomicU64,
}

pub(super) struct Before {
    id: u64,
    began: Instant,
    queue_us: Option<u128>,
    probe_us: u128,
    physical: Option<u64>,
    residency: Result<serde_json::Value, String>,
    priority: Priority,
    access: Access,
    tid: i64,
}

impl Trace {
    pub(super) fn new(files: &[ModelFile]) -> Self {
        let trace = Self {
            id: NEXT_SOURCE.fetch_add(1, Ordering::Relaxed),
            epoch: Instant::now(),
            unix_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            owned: files
                .iter()
                .map(|file| {
                    file.0
                        .metadata()
                        .is_ok_and(|m| m.uid() == unsafe { libc::geteuid() })
                })
                .collect(),
            next_read: AtomicU64::new(1),
        };
        for (index, file) in files.iter().enumerate() {
            let metadata = file.0.metadata().ok();
            eprintln!(
                "expert source pages {}",
                serde_json::json!({
                    "schema": 1, "kind": "file", "source": trace.id, "file": index,
                    "path": std::fs::read_link(format!("/proc/self/fd/{}", file.0.as_raw_fd())).ok(),
                    "bytes": metadata.as_ref().map(|m| m.len()),
                    "device": metadata.as_ref().map(|m| m.dev()),
                    "inode": metadata.as_ref().map(|m| m.ino()),
                    "owned_by_process_user": trace.owned[index],
                    "epoch_unix_ns": trace.unix_ns,
                    "limits": "Observation-only mincore mappings; per-thread physical I/O; instrumented service, not a speed measurement",
                })
            );
        }
        trace
    }

    pub(super) fn before(
        &self,
        file: &File,
        range: &Range,
        priority: Priority,
        access: Access,
        submitted: Option<Instant>,
    ) -> Before {
        let probe_began = Instant::now();
        let residency = if self.owned[range.file] {
            residency(file, range)
        } else {
            Err("mincore may censor residency for a file owned by another user".into())
        };
        let physical = thread_read_bytes();
        let began = Instant::now();
        Before {
            id: self.next_read.fetch_add(1, Ordering::Relaxed),
            began,
            queue_us: submitted.map(|s| probe_began.saturating_duration_since(s).as_micros()),
            probe_us: began.saturating_duration_since(probe_began).as_micros(),
            physical,
            residency,
            priority,
            access,
            tid: unsafe { libc::syscall(libc::SYS_gettid) } as i64,
        }
    }

    pub(super) fn finish(
        &self,
        before: Before,
        range: &Range,
        success: bool,
        host_cache_hit: bool,
    ) -> serde_json::Value {
        let ended = Instant::now();
        let probe_began = Instant::now();
        let physical = thread_read_bytes()
            .zip(before.physical)
            .and_then(|(after, prior)| after.checked_sub(prior));
        let after_probe_us = probe_began.elapsed().as_micros();
        let (residency, error) = match before.residency {
            Ok(value) => (Some(value), None),
            Err(error) => (None, Some(error)),
        };
        serde_json::json!({
            "schema": 1, "kind": "read", "source": self.id, "read": before.id,
            "tid": before.tid, "file": range.file, "offset": range.offset,
            "bytes": range.bytes, "success": success, "host_cache_hit": host_cache_hit,
            "priority_at_start": if before.priority == Priority::Demand { "demand" } else { "prefetch" },
            "access": if before.access == Access::Demand { "demand" } else { "bulk" },
            "begin_unix_ns": self.unix_ns + before.began.saturating_duration_since(self.epoch).as_nanos(),
            "end_unix_ns": self.unix_ns + ended.saturating_duration_since(self.epoch).as_nanos(),
            "queue_us": before.queue_us,
            "service_us": ended.saturating_duration_since(before.began).as_micros(),
            "before_probe_us": before.probe_us, "after_probe_us": after_probe_us,
            "thread_physical_read_bytes": physical,
            "before": residency, "probe_error": error,
        })
    }
}

fn thread_read_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/thread-self/io")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes:")?.trim().parse().ok())
}

fn residency(file: &File, range: &Range) -> Result<serde_json::Value, String> {
    static PAGE: OnceLock<usize> = OnceLock::new();
    let page = *PAGE.get_or_init(|| {
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(size).unwrap_or(4096).max(1)
    });
    let start = range.offset / page as u64 * page as u64;
    let head = usize::try_from(range.offset - start).map_err(|e| e.to_string())?;
    let needed = head
        .checked_add(range.bytes)
        .ok_or("page cover overflows")?;
    if needed == 0 {
        return Err("empty source range".into());
    }
    let pages = needed.div_ceil(page);
    let length = pages.checked_mul(page).ok_or("page cover overflows")?;
    let offset = libc::off_t::try_from(start).map_err(|e| e.to_string())?;
    // PROT_NONE, without MAP_POPULATE: the mapping does not fault payload pages.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_NONE,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            offset,
        )
    };
    if mapping == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let mut flags = vec![0u8; pages];
    let status = unsafe { libc::mincore(mapping, length, flags.as_mut_ptr()) };
    let error = (status != 0).then(|| std::io::Error::last_os_error().to_string());
    let unmap = unsafe { libc::munmap(mapping, length) };
    if let Some(error) = error {
        return Err(error);
    }
    if unmap != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let mut resident_pages = 0usize;
    let mut resident_bytes = 0usize;
    for (index, flag) in flags.into_iter().enumerate() {
        if flag & 1 != 0 {
            resident_pages += 1;
            let low = (index * page).max(head);
            let high = ((index + 1) * page).min(needed);
            resident_bytes += high.saturating_sub(low);
        }
    }
    Ok(serde_json::json!({"page_bytes": page, "cover_pages": pages,
        "resident_pages": resident_pages, "resident_requested_bytes": resident_bytes,
        "missing_requested_bytes": range.bytes - resident_bytes}))
}
