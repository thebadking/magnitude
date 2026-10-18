//! The file source (spec 5.1): stored records read straight from the model
//! file into staging by persistent workers. macOS reads bypass the page cache;
//! other Unix platforms currently use buffered reads with random-access advice.
//!
//! Reads come in two classes. A demand read is one a parked step waits on; a
//! prefetch is a prediction. Workers always take a queued demand read before
//! any prefetch, a prefetch that has not started can be cancelled, and a
//! queued prefetch can be promoted to demand when a step turns out to need it.
//! Without demand priority and cancellation, wrong prefetches delay needed
//! reads and prefetching is slower than none (theory §3, §6).

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};
use super::cache::{Access, HostExpertCache};

/// Persistent workers. More did not raise bandwidth on any machine measured
/// (theory §2).
pub(crate) const WORKERS: usize = 8;

/// Reads around the page cache take their fast path only for block-aligned
/// file offsets, lengths and buffers; anything else falls back to a slow
/// path (about 50 MB/s against 2.45 GB/s on an M4 Pro instance). A worker
/// reads a range's aligned cover into its own aligned buffer and copies the
/// range's exact bytes into staging.
#[cfg(target_os = "macos")]
const BLOCK: usize = 16384;

/// A worker's block-aligned read buffer, grown to the largest cover read.
#[cfg(target_os = "macos")]
struct AlignedBuffer {
    base: Option<std::ptr::NonNull<u8>>,
    capacity: usize,
}

#[cfg(target_os = "macos")]
impl AlignedBuffer {
    fn new() -> Self {
        Self {
            base: None,
            capacity: 0,
        }
    }

    fn layout(capacity: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(capacity, BLOCK).expect("a block-aligned layout")
    }

    /// At least `bytes` aligned bytes.
    fn get(&mut self, bytes: usize) -> &mut [u8] {
        if bytes > self.capacity {
            self.release();
            let layout = Self::layout(bytes);
            // A nonzero, block-aligned layout.
            let base = unsafe { std::alloc::alloc(layout) };
            self.base = Some(
                std::ptr::NonNull::new(base)
                    .unwrap_or_else(|| std::alloc::handle_alloc_error(layout)),
            );
            self.capacity = bytes;
        }
        let base = self.base.expect("allocated above");
        // `capacity >= bytes` bytes owned by this buffer.
        unsafe { std::slice::from_raw_parts_mut(base.as_ptr(), bytes) }
    }

    fn release(&mut self) {
        if let Some(base) = self.base.take() {
            // Allocated with this layout.
            unsafe { std::alloc::dealloc(base.as_ptr(), Self::layout(self.capacity)) };
        }
        self.capacity = 0;
    }
}

#[cfg(target_os = "macos")]
impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        self.release();
    }
}

/// Read `range` through its block-aligned cover. The cover may run past the
/// end of the file; only the range's own bytes must be read.
#[cfg(target_os = "macos")]
fn read_range(file: &File, range: &Range, buffer: &mut AlignedBuffer) -> std::io::Result<()> {
    let block = BLOCK as u64;
    let start = range.offset / block * block;
    let head = (range.offset - start) as usize;
    let needed = head + range.bytes;
    let cover = needed.div_ceil(BLOCK) * BLOCK;
    let buffer = buffer.get(cover);
    let mut done = 0;
    while done < needed {
        match file.read_at(&mut buffer[done..cover], start + done as u64) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(read) => done += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    // The destination is staging reserved for this record until its ticket
    // ends.
    let destination = unsafe { std::slice::from_raw_parts_mut(range.destination, range.bytes) };
    destination.copy_from_slice(&buffer[head..needed]);
    Ok(())
}

/// Buffered reads have no destination-alignment requirement. The ticket owns
/// this staging range until completion, so the kernel can copy exactly the
/// requested bytes into it without an aligned cover or an intermediate copy.
#[cfg(not(target_os = "macos"))]
fn read_range(file: &File, range: &Range) -> std::io::Result<()> {
    let destination = unsafe { std::slice::from_raw_parts_mut(range.destination, range.bytes) };
    file.read_exact_at(destination, range.offset)
}

/// The model file, opened with the platform's expert-read policy.
pub(crate) struct ModelFile(File, Option<(Arc<HostExpertCache>, usize)>);

impl ModelFile {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        configure_reads(&file).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Self(file, None))
    }

    pub(crate) fn with_cache(mut self, path: &Path, cache: Arc<HostExpertCache>) -> Self {
        let file = cache.register_file(path);
        self.1 = Some((cache, file)); self
    }

    /// Successful explicit retention replaces, rather than duplicates, the
    /// reclaimable file cache. Only whole pages inside this record are hinted;
    /// adjacent records and active mapped readers keep their own lifetimes.
    fn release_cached_file_pages(&self, range: &Range) {
        #[cfg(target_os = "linux")]
        {
            static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let page = *PAGE.get_or_init(|| {
                let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
                u64::try_from(size).unwrap_or(4096).max(1)
            });
            let start = range.offset.div_ceil(page) * page;
            let end = (range.offset + range.bytes as u64) / page * page;
            if end > start && end <= i64::MAX as u64 {
                // A cache hint cannot fail a completed expert fetch.
                unsafe { libc::posix_fadvise(self.0.as_raw_fd(), start as i64,
                    (end - start) as i64, libc::POSIX_FADV_DONTNEED); }
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = range;
    }
}

#[cfg(target_os = "macos")]
fn configure_reads(file: &File) -> std::io::Result<()> {
    // Faulting experts in through the page cache is the 3 tok/s case
    // (spec §3 finding 5).
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn configure_reads(file: &File) -> std::io::Result<()> {
    // Linux page-cache advice for a sequential-random reader; O_DIRECT needs
    // aligned destinations the staging layout does not promise.
    let status =
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status));
    }
    Ok(())
}

/// One contiguous range of the model file into host-addressed staging.
pub(crate) struct Range {
    pub file: usize,
    pub offset: u64,
    pub bytes: usize,
    pub destination: *mut u8,
}

// The destination is staging the source's owner keeps alive and does not
// touch until the read completes.
unsafe impl Send for Range {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Priority {
    Demand,
    Prefetch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Every range has its bytes.
    Done,
    /// A read failed; the staging record holds nothing usable.
    Failed,
    /// Cancelled before any range started.
    Cancelled,
}

/// A record's reads: complete when every range is.
pub(crate) struct Ticket {
    state: Mutex<TicketState>,
    changed: Condvar,
}

struct TicketState {
    /// Ranges still queued.
    queued: usize,
    /// Ranges still being read.
    running: usize,
    failed: bool,
    error: Option<String>,
    cancelled: bool,
    priority: Priority,
    progress_at: Instant,
}

impl Ticket {
    #[cfg(test)]
    pub(crate) fn controlled() -> Arc<Self> {
        Self::controlled_ranges(1)
    }
    #[cfg(test)]
    pub(crate) fn controlled_ranges(ranges: usize) -> Arc<Self> {
        let ticket = Self::done(Priority::Demand);
        ticket.state.lock().unwrap().running = ranges;
        ticket
    }
    #[cfg(test)]
    pub(crate) fn complete_controlled_range(&self) {
        let mut state = self.state.lock().unwrap();
        state.running -= 1;
        state.progress_at = Instant::now();
        self.changed.notify_all();
    }
    #[cfg(test)]
    pub(crate) fn finish_controlled(&self, failed: bool) {
        let mut state = self.state.lock().unwrap();
        state.running = 0; state.failed = failed; state.progress_at = Instant::now();
        self.changed.notify_all();
    }
    /// A record whose bytes are already in staging: no read.
    pub(crate) fn done(priority: Priority) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(TicketState {
                queued: 0,
                running: 0,
                failed: false,
                error: None,
                cancelled: false,
                priority,
                progress_at: Instant::now(),
            }),
            changed: Condvar::new(),
        })
    }

    fn outcome(state: &TicketState) -> Option<Outcome> {
        if state.cancelled {
            return Some(Outcome::Cancelled);
        }
        (state.queued == 0 && state.running == 0).then_some(if state.failed {
            Outcome::Failed
        } else {
            Outcome::Done
        })
    }

    /// The outcome, if the reads have ended.
    pub(crate) fn poll(&self) -> Option<Outcome> {
        Self::outcome(&self.state.lock().expect("ticket lock"))
    }

    /// Latest range start or completion, independent of its queue priority.
    pub(crate) fn progress_at(&self) -> Instant {
        self.state.lock().expect("ticket lock").progress_at
    }

    /// Snapshot for an unsuccessful fetch, including outstanding work.
    pub(crate) fn diagnostic(&self) -> String {
        let state = self.state.lock().expect("ticket lock");
        format!("outcome={:?}, queued={}, running={}, idle_ms={}, error={}",
            Self::outcome(&state), state.queued, state.running,
            state.progress_at.elapsed().as_millis(), state.error.as_deref().unwrap_or("none"))
    }

    /// Wait for the reads to end, at most until `deadline`.
    pub(crate) fn wait_until(&self, deadline: Instant) -> Option<Outcome> {
        let mut state = self.state.lock().expect("ticket lock");
        loop {
            if let Some(outcome) = Self::outcome(&state) {
                return Some(outcome);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            state = self
                .changed
                .wait_timeout(state, deadline - now)
                .expect("ticket lock")
                .0;
        }
    }

    pub(crate) fn priority(&self) -> Priority {
        self.state.lock().expect("ticket lock").priority
    }
}

struct Job {
    ticket: Arc<Ticket>,
    range: Range,
    access: Access,
}

struct Queues {
    demand: VecDeque<Job>,
    prefetch: VecDeque<Job>,
    stopping: bool,
}

struct Shared {
    files: Vec<ModelFile>,
    queues: Mutex<Queues>,
    work: Condvar,
    cache_hit_bytes: AtomicU64,
    file_read_bytes: AtomicU64,
}

pub(crate) struct ExpertSource {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl ExpertSource {
    pub(crate) fn new(files: Vec<ModelFile>) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            files,
            queues: Mutex::new(Queues {
                demand: VecDeque::new(),
                prefetch: VecDeque::new(),
                stopping: false,
            }),
            work: Condvar::new(),
            cache_hit_bytes: AtomicU64::new(0),
            file_read_bytes: AtomicU64::new(0),
        });
        let workers = (0..WORKERS)
            .map(|index| {
                let shared = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("expert-source-{index}"))
                    .spawn(move || work(&shared))
                    .map_err(|error| format!("expert source worker: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { shared, workers })
    }

    /// Queue a record's ranges.
    pub(crate) fn submit(&self, ranges: Vec<Range>, priority: Priority) -> Arc<Ticket> {
        self.submit_with_access(ranges, priority, if priority == Priority::Demand { Access::Demand } else { Access::Bulk })
    }

    pub(crate) fn submit_bulk(&self, ranges: Vec<Range>, priority: Priority) -> Arc<Ticket> {
        self.submit_with_access(ranges, priority, Access::Bulk)
    }

    pub(crate) fn counters(&self) -> (u64, u64) {
        (self.shared.cache_hit_bytes.load(Ordering::Relaxed), self.shared.file_read_bytes.load(Ordering::Relaxed))
    }

    pub(crate) fn device_residency(&self, tensor: super::runtime::StoredTensor, expert: usize, resident: bool) {
        if let Some((cache, file)) = &self.shared.files[tensor.file].1 {
            cache.device_residency(*file, tensor.base + expert as u64 * tensor.bytes,
                tensor.bytes as usize, resident);
        }
    }

    fn submit_with_access(&self, ranges: Vec<Range>, priority: Priority, access: Access) -> Arc<Ticket> {
        let ticket = Arc::new(Ticket {
            state: Mutex::new(TicketState {
                queued: ranges.len(),
                running: 0,
                failed: false,
                error: None,
                cancelled: false,
                priority,
                progress_at: Instant::now(),
            }),
            changed: Condvar::new(),
        });
        let mut queues = self.shared.queues.lock().expect("source queues");
        let queue = match priority {
            Priority::Demand => &mut queues.demand,
            Priority::Prefetch => &mut queues.prefetch,
        };
        for range in ranges {
            queue.push_back(Job {
                ticket: Arc::clone(&ticket),
                range,
                access,
            });
        }
        drop(queues);
        self.shared.work.notify_all();
        ticket
    }

    /// A prefetch a parked step now needs: its queued ranges move to the
    /// demand queue, ahead of every other prefetch.
    pub(crate) fn promote(&self, ticket: &Arc<Ticket>) {
        let mut queues = self.shared.queues.lock().expect("source queues");
        ticket.state.lock().expect("ticket lock").priority = Priority::Demand;
        let mut kept = VecDeque::with_capacity(queues.prefetch.len());
        while let Some(job) = queues.prefetch.pop_front() {
            if Arc::ptr_eq(&job.ticket, ticket) {
                queues.demand.push_back(job);
            } else {
                kept.push_back(job);
            }
        }
        queues.prefetch = kept;
    }

    /// Cancel a prefetch none of whose ranges has started; true when
    /// cancelled. A started one runs to the end.
    pub(crate) fn cancel(&self, ticket: &Arc<Ticket>) -> bool {
        let mut queues = self.shared.queues.lock().expect("source queues");
        let mut state = ticket.state.lock().expect("ticket lock");
        if state.running > 0 || state.cancelled || Ticket::outcome(&state).is_some() {
            return state.cancelled;
        }
        queues
            .prefetch
            .retain(|job| !Arc::ptr_eq(&job.ticket, ticket));
        queues.demand.retain(|job| !Arc::ptr_eq(&job.ticket, ticket));
        state.queued = 0;
        state.cancelled = true;
        drop(state);
        ticket.changed.notify_all();
        true
    }
}

impl Drop for ExpertSource {
    fn drop(&mut self) {
        self.shared.queues.lock().expect("source queues").stopping = true;
        self.shared.work.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn work(shared: &Shared) {
    #[cfg(target_os = "macos")]
    let mut buffer = AlignedBuffer::new();
    loop {
        let job = {
            let mut queues = shared.queues.lock().expect("source queues");
            loop {
                if let Some(job) = queues.demand.pop_front().or_else(|| queues.prefetch.pop_front()) {
                    let mut state = job.ticket.state.lock().expect("ticket lock");
                    state.queued -= 1;
                    state.running += 1;
                    state.progress_at = Instant::now();
                    drop(state);
                    break job;
                }
                if queues.stopping {
                    return;
                }
                queues = shared.work.wait(queues).expect("source queues");
            }
        };
        let file = &shared.files[job.range.file];
        let cached = file.1.as_ref().and_then(|(cache, id)|
            cache.get(*id, job.range.offset, job.range.bytes, job.access));
        let result = if let Some(record) = cached {
            // The ticket still exclusively owns the destination. The Arc
            // keeps these immutable bytes alive across eviction and copying.
            let destination = unsafe { std::slice::from_raw_parts_mut(job.range.destination, job.range.bytes) };
            destination.copy_from_slice(record.bytes());
            shared.cache_hit_bytes.fetch_add(job.range.bytes as u64, Ordering::Relaxed);
            Ok(())
        } else {
            #[cfg(target_os = "macos")]
            let result = read_range(&file.0, &job.range, &mut buffer);
            #[cfg(not(target_os = "macos"))]
            let result = read_range(&file.0, &job.range);
            if result.is_ok() {
                shared.file_read_bytes.fetch_add(job.range.bytes as u64, Ordering::Relaxed);
                if let Some((cache, id)) = &file.1 {
                    // Retention finishes before the ticket publishes: its
                    // staging destination cannot be reused during this copy.
                    let bytes = unsafe { std::slice::from_raw_parts(job.range.destination, job.range.bytes) };
                    if cache.remember(*id, job.range.offset, bytes, job.access) {
                        file.release_cached_file_pages(&job.range);
                    }
                }
            }
            result
        };
        let mut state = job.ticket.state.lock().expect("ticket lock");
        state.running -= 1;
        state.progress_at = Instant::now();
        if let Err(error) = result {
            state.failed = true;
            state.error.get_or_insert_with(|| format!(
                "file {}, offset {}, bytes {}: {error}",
                job.range.file, job.range.offset, job.range.bytes));
        }
        drop(state);
        job.ticket.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn one_ticket_reads_identical_offsets_from_distinct_shards() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("magnitude-source-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let paths = [directory.join("first.gguf"), directory.join("second.gguf")];
        std::fs::write(&paths[0], b"aaaFIRSTzzz").unwrap();
        std::fs::write(&paths[1], b"bbbOTHERyyy").unwrap();
        let source = ExpertSource::new(
            paths
                .iter()
                .map(|path| ModelFile::open(path).unwrap())
                .collect(),
        )
        .unwrap();
        let mut first = [0u8; 5];
        let mut second = [0u8; 5];
        let ticket = source.submit(
            vec![
                Range {
                    file: 1,
                    offset: 3,
                    bytes: 5,
                    destination: second.as_mut_ptr(),
                },
                Range {
                    file: 0,
                    offset: 3,
                    bytes: 5,
                    destination: first.as_mut_ptr(),
                },
            ],
            Priority::Demand,
        );
        let outcome = ticket.wait_until(Instant::now() + Duration::from_secs(1));
        // Join readers before checking, including the timeout failure path.
        drop(source);
        assert_eq!(outcome, Some(Outcome::Done));
        assert_eq!(&first, b"FIRST");
        assert_eq!(&second, b"OTHER");
        std::fs::remove_dir_all(directory).unwrap();
    }
}
