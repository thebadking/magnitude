# Magnitude inference research and implementation stages

Updated: 2026-10-19

## Objective

Improve generation latency on the current Qwen3.8 Flash Next deployment while
preserving correctness, memory safety, context capacity, and the existing
single-GPU paths. Measure prefill, generation, and complete-request performance
separately. A faster prompt does not establish faster generation.

Implement and evaluate one stage at a time. Retain an improvement only when its
benefit is repeatable on the intended workload. Record unsuccessful experiments
and keep useful experimental controls opt-in.

This is a research and execution plan. Proposed techniques and acceptance bars
are not measured speedups. The applicable documents under `design/` remain the
architecture authority and must be consulted before implementation changes.

## Current system and evidence

### Target environment

| Item | Observed configuration |
| --- | --- |
| CPU | AMD Ryzen 9 5900X, 12 cores; AVX2 available; AVX-512 and AMX unavailable |
| RAM | Marketed 64 GB; approximately 62.7 GiB OS-usable |
| GPUs | Two RTX 3090 Ti cards, 24 GB each; target CUDA capability SM86 |
| GPU communication | Peer reads report unsupported; current boundary transport uses host staging |
| Model | Catalog reference `qwen3.8-flash-next:gguf:q4`, UD-Q4_K_XL package |
| Model size | Four GGUF shards, approximately 103.7 GiB total |
| Relevant geometry | 48 decoder blocks, four residual streams; the main routed bank has 512 experts with 10 selected; record exact per-layer bindings for each experiment |
| Placement | Explicit contiguous ranges, currently decoder blocks 0–23 and 24–47 |
| Default generation | Plain decoding, lookahead enabled, prompt group 8 |
| Managed RAM expert cache | Opt-in; unset/zero disables, `MAGNITUDE_HOST_EXPERT_CACHE_BYTES=auto` enables automatic capacity |

Do not treat RAM and VRAM as one interchangeable allocation pool. Runtime state,
staging, representations, duplicated holdings, and each pool's reserves consume
capacity too. Process RSS includes mapped file pages and excludes additional
kernel file cache.

### Historical controlled comparison

The 2026-10-18 comparison used one canonical prose-repeat fixture, 14,967 input
tokens, 256 generated tokens, and two passes per condition. All six warmups and
six measured responses passed exact-prefix checks. MTP and retained-prefix reuse
were disabled. The order was cache-on / cache-off / llama.cpp / llama.cpp /
cache-off / cache-on under one machine lease and thermal gate.

| Condition | Mean prefill | Mean generation | Mean total model time | Mean generation rate |
| --- | ---: | ---: | ---: | ---: |
| Magnitude, managed cache off | 68.00 s | 24.60 s | 92.60 s | 10.43 tokens/s |
| llama.cpp | 86.99 s | 22.90 s | 109.89 s | 11.18 tokens/s |
| Magnitude, managed cache automatic | 98.12 s | 26.70 s | 124.82 s | 9.61 tokens/s |

Magnitude with the managed cache disabled used 15.7% less total model time, with
6.7% lower generation throughput. The managed cache's prefill times varied from
135.03 s to 61.21 s, so its average needs that context. This experiment did not
justify enabling it by default. Linux buffered file service already benefits
from RAM through the operating system's file cache.

The comparison used Magnitude's default KV codec and llama.cpp with 26 GPU layers
and f16 KV storage. It is a comparison of those serving configurations, not an
operator comparison with identical representations. Loading and warmups are
excluded from model-time figures. Rates are means of individual request rates.

Evidence, available locally under
`inference/evaluation/results/20261018T194603Z-host-cache-comparison-8c3f34/`:

- [Timing summary](inference/evaluation/results/20261018T194603Z-host-cache-comparison-8c3f34/timing-summary.json)
- [Build identity](inference/evaluation/results/20261018T194603Z-host-cache-comparison-8c3f34/build.json)
- [Configuration](inference/evaluation/results/20261018T194603Z-host-cache-comparison-8c3f34/comparison.json)
- [Output checks](inference/evaluation/results/20261018T194603Z-host-cache-comparison-8c3f34/prefix-checks.json)

Those artifacts are ignored local files, not bundled benchmark results for every
clone. The measured build was `9eca152d` plus dirty digest
`a742b742257ae6c1bb9298fe94bdda0bb71a840d5926535e6c094ed5c5496ab8`.
The subsequent cache commit is `91c753b3`; the checkout reviewed for this plan
also contains `5cac0a6f`, an ahead-layer norm-port fix. Existing timings are not
a fresh measurement of that latest checkout. Preserve that existing commit.

The earlier context-capacity run qualified a configured limit of 262,144 tokens
and an actual 260,137-input + 37-output retrieval request. That result predates
the cache work. Requalify affected state/memory paths when a stage warrants it;
do not repeat a full maximum-context run for every unrelated change.

### RAM occupancy and actual disk traffic

The machine already uses most spare RAM for cached model pages. During the
Stage 2a tool qualification, `free` showed approximately 3.2 GiB actually free
and 54 GiB in buffers/cache. These are system totals, not the engine's RSS.
The roughly 54 GiB reported as **available** includes reclaimable file cache;
allocating it to an explicit cache would reclaim existing cached pages.
[Linux documents this distinction](https://docs.kernel.org/filesystems/proc.html#meminfo).

A read-only `mincore` inventory after that job found **52.580 GiB of the
103.688 GiB GGUF shards resident in RAM**. The probe mapped files with
`PROT_NONE`, queried page residency and unmapped them; it read no model payloads
and neither warmed nor dropped file pages. This is a changing, post-run snapshot,
not an expert hot-set inventory or a timed comparison. It establishes that model
bytes already occupy substantial RAM even with the managed cache disabled.
The model alone exceeds host RAM; useful selection matters more than allocating
another anonymous copy of the same records. GPU-native representations and
runtime reserves also prevent treating RAM plus VRAM as one simple capacity sum.

Disk remains relevant when requested pages are absent, especially during loading
and broad prompt scans. During the controlled Stage 2a measured generation
windows, however, original-policy logical reads were about 12.963–12.981 GB
while process-attributed physical reads were only 3.064–22.544 MB. Candidate
windows read about 6.928–6.934 GB logically and 2.957–271.303 MB physically.
These delivery windows may include lookahead and are not exact device-step
boundaries. They show why a file-read byte counter alone must not be labeled SSD
traffic, and why warm generation can still benefit from avoiding RAM copies,
host-to-device uploads and conversions.

Retain the managed cache as opt-in after its historical comparison above.
Any later RAM-cache revision must reduce physical misses or source service time
without losing more useful Linux cache or adding costly copies. Record file
residency, logical/physical reads, swap activity and latency together. High
`MemAvailable` alone is not evidence of an unused RAM opportunity.

Local evidence: `20261019T075917Z-stage2a-comparison-b3e32b/` contains
`ram-residency.json`, its reproduction script, measured I/O windows and
`memory-summary.json`. The residency snapshot was taken after qualification;
it did not interfere with timed A/B/B/A requests.

A fresh read-only check at **2026-10-19 09:22 UTC**, with no inference process
running and both GPUs idle, found the following:

| Host RAM observation | GiB |
| --- | ---: |
| OS-usable total | 62.696 |
| Actually free | 6.892 |
| Linux `Cached` | 53.508 |
| `MemAvailable`, including reclaimable cache | 60.664 |
| Main GGUF pages resident, measured with `mincore` | 46.313 |

This idle snapshot does not establish peak inference occupancy. It confirms that
the apparent 60.7 GiB available is largely RAM already serving as file cache.
The remaining free RAM can absorb additional pages, subject to runtime reserves;
an explicit cache also competes with existing cached pages. More useful retention
can help cold reads and broad prompt scans, but simply allocating more RAM has
not demonstrated a warm-generation win. In the Stage 2b candidate's measured
generation delivery windows, roughly 4.84 GB of logical reads corresponded to
only 6.14 MB of process-attributed physical reads. Investigate RAM copies,
host-to-device uploads, conversion and expert reuse alongside physical misses;
these counters alone do not isolate which remaining operation dominates latency.
The managed-cache historical comparison above remains the reason it is opt-in.

Fresh snapshot and its non-warming reproduction script are under
`inference/evaluation/results/20261019T092219Z-idle-ram-diagnostic/`.

### Confirmed code observations

- [CUDA upload](inference/seismic/backends/cuda/src/driver.rs) calls
  `cuMemcpyHtoDAsync` and then synchronizes the transfer stream before returning.
  The [expert streaming runtime](inference/engine/executor/src/streaming/runtime.rs)
  invokes this path separately for each fetched expert tensor.
- [Placed startup](inference/engine/src/execution/placed.rs) currently requires
  two ranges and plain text execution. Prediction-head support exists in
  [the Qwen4 family](inference/engine/families/qwen4exp/src/head.rs), but that does
  not establish MTP support across the two owners.
- [Expert prediction](inference/engine/executor/src/operators/routed/fused_graph.rs)
  is disabled for mixed or unnormalized residual forms. The previous predictor
  wasted source traffic on this model; re-enabling it unchanged is not a plan.
- [GPU expert replacement](inference/engine/executor/src/streaming/arena.rs)
  primarily uses recency among eligible, unleased slots.
- CUDA graphs, quantized KV storage, prefix reuse, lookahead, fused quantized
  kernels, and grouped prompt processing already exist. Research should improve
  their composition with streamed experts.

## Method for every stage

1. State the hypothesis, affected path, primary metric, and smallest experiment
   that can decide it. Identify the applicable design documents with
   `bun design-docs <paths>` before editing implementation files.
2. Preserve the last accepted implementation and its executable/build identity.
   Change one mechanism at a time; give each independent mechanism its own
   measured decision. Do not bundle several optimizations behind one result.
3. Complete the implementation and release build before model qualification.
   Keep source and executables frozen while a model run is live.
4. Run the smallest relevant correctness check. Then compare baseline A and
   candidate B in an alternating A/B/B/A session. Start with two measured
   requests per side; add paired observations only if noise prevents a decision.
5. Retain, revise, revert, or defer the candidate explicitly. A failed candidate
   does not become the next baseline. Update this document's experiment ledger.
6. Commit a retained implementation with its relevant design updates. Preserve
   raw failures, reproduction information, and the reason for the decision.
   Advance to the next stage only after disposing of the current experiment.

### Benchmark controls

- Use `inference/evaluation`, catalog acquisition, canonical chat fixtures, and
  the engine's own renderer. Follow [the evaluation contract](design/inference/evaluation.md).
  Never substitute raw text, random tokens, or fabricated token counts.
- Run through `magnitude_remote.flow.Job` / `execute`, the canonical machine
  lock, and the thermal gate for timed comparisons. Select a pool unless this
  particular two-GPU host is required. Bound the lease and request durations.
- Record commit and dirty digest, executable hashes, model/shard identity,
  rendered input identity, hardware/topology, driver and engine versions,
  tuning state/budget, and all runtime options for both sides.
- Keep placement, prompt grouping, KV formats, context, output budget, sampling,
  prefix-cache policy, and managed expert-cache policy fixed unless one is the
  declared experimental variable. Use greedy sampling and seed 42 where supported.
- Distinguish compiled-kernel/tuning caches, OS/managed expert caches, and retained
  conversation-prefix state. A fresh process does not imply cold SSD reads.
  Do not drop the host's global file cache to manufacture a comparison.
- Give each launch the same declared warmup. Exclude loading, warmup, and
  qualification from steady-state model timings. Report startup separately when
  a change affects it. Keep profiling instrumentation out of final timing runs.
- Check actual prompt/output counts. Exact-copy/retrieval fixtures require exact
  answers; other fixtures use their canonical semantic checks. Do not equate
  different generated work with identical executor performance. Do not relax
  validation to admit a timing. Keep invalid observations in outcome counts.
- Compare a candidate to the last accepted Magnitude build first. Use the frozen
  llama.cpp configuration as an independent reference when relevant; compare
  against a separately tuned reference configuration before broad performance
  claims. Record supported flags rather than assuming a local binary's features.
- Separate plain-vs-plain implementation comparisons from speculative-vs-plain
  product comparisons. A speculative configuration must report actual proposals
  and accepted tokens. Preserve target sampling semantics and existing precision policy.

### Workloads, metrics, and promotion

Use a small fixture for the first correctness decision, then the workload most
affected by the stage. Broaden coverage after a candidate shows value; avoid
paying for a full matrix before knowing whether it helps.

| Workload | Purpose |
| --- | --- |
| Short prompt with sustained generation | Generation latency and transfer/cache behavior |
| Established 14,967 + 256 fixture | Continuity with the historical long-prompt comparison |
| Canonical code and tool fixtures | Representative agent work and semantic correctness |
| Prefix reuse, cancellation, and recovery | State/lease lifecycle for affected paths |
| Long-context retrieval | Context and memory qualification when a stage changes those paths |
| Multiple independent sequences | Aggregate throughput for scheduling/placement stages, reported separately |

Record model prefill/generation time, time to first token, completed-request
latency, actual tokens, and individual-run spread. Record GPU allocation/census
and headroom, host memory/census and RSS, physical disk reads, logical source
reads, expert hit/miss rates, and transfer bytes when applicable. For MTP, include
draft, verification, repair time, proposal count, acceptance, and committed tokens.

The working practical bar for default promotion is at least 3% improvement in
the declared primary metric **and** a difference beyond observed measurement
noise. A smaller change may be retained for another demonstrated engineering
benefit, but receives no speedup claim. Treat a measured regression above 5% in
prefill or complete-request time as a material tradeoff requiring an explicit
workload-specific profile rather than a silent default change. Any correctness
failure or memory-reserve violation blocks promotion. These percentages are
decision bars, not predictions or statistical confidence estimates.

Do not infer general model quality from a repeated passage. Mathematical/kernel
changes require the applicable numerical qualification. Quality comparisons
against a reference require that reference to pass its sanity gate.

## Stages

| Stage | Work | Current status | Depends on |
| --- | --- | --- | --- |
| 0 | Current baseline and critical-path measurement | Baseline established; attribution limits recorded | Current checkout |
| 1 | Batched asynchronous expert transfers | Disposed: prompt candidate failed; decode candidate not promoted | 0 |
| 2 | Trace-informed GPU expert retention | 2a generation bar passed; tool gate and optional-profile decision next | 1 decision |
| 3 | Confidence-limited expert prefetch | Pending | 2 decision |
| 4 | MTP across the two owners | Pending | 3 decision |
| 5 | SM86 packed-kernel and representation improvements | Pending | 4 decision |
| 6 | Selective CPU expert execution | Exploratory | Measured transfer/compute crossover |
| 7 | Adaptive placement and prompt grouping | Pending | Accepted earlier stages |
| 8 | Consolidated qualification and reference comparison | Pending | Selected candidates disposed |

A dependency on a decision permits a recorded revert or deferral. It does not
require forcing an unsuccessful optimization into the code. Reorder remaining
stages only with a recorded reason from new evidence.

### Stage 0 — Establish the current baseline and locate exposed waits

**Hypothesis:** Some generation latency is exposed expert service or transfer
waiting; the previous aggregate timing does not identify its exact share.

- Preserve/rebuild the latest source as needed and verify executable provenance.
- Obtain a current plain-decoding baseline with managed retention disabled.
  Reuse historical results as context, not as a same-session control.
- Attribute time to file service, RAM copies, H2D transfers, conversion, resident
  expert math, attention/readout, and inter-owner boundary/host coordination.
- Separate physical SSD reads from OS-cache hits and per-layer cache misses.
  Correlate waits with completed work; overlapping durations cannot simply be
  summed into a wall-time breakdown.
- Prefer existing counters and bounded diagnostic tracing. Check whether added
  instrumentation changes graph execution or substantially perturbs timing.

**Benchmark:** Start with one short diagnostic and one necessary matched timing
case. Add another fixture only to resolve an observed ambiguity.

**Exit:** A reproducible baseline, preserved executable identity, and enough
evidence to select Stage 1's batch/completion strategy or record a better next target.

### Stage 1 — Batch transfers and complete them with events

**Hypothesis:** Repeated transfer-stream synchronization adds avoidable host
serialization and prevents overlap with other ready source work.

- Add an asynchronous protocol-transfer operation with an owned completion
  handle. Reuse suitable existing CUDA stream/event and pinned-memory machinery.
- Enqueue gate/up/down and other already-ready transfers together. Do not wait
  for unrelated unfinished reads merely to form a larger batch.
- Publish an expert's readiness only after its required copies complete. Retain
  source tickets, pinned storage, and destination leases until their final users
  finish. Preserve cancellation, progress/watchdogs, and failure propagation.
- Preserve the parked-kernel publication protocol; an ordinary allocation fence
  must not wait on the kernel that is itself waiting for these copies.

**Benchmark:** A/B/B/A plain decoding with unchanged expert policy and kernels.
Measure synchronization count, transfer time, exposed wait, and generation
latency. Qualify cancellation/recovery and allocation lifetime before promotion.

**Exit:** Correct completion/lifetime behavior and a measured decision on the
latency benefit. Fewer API calls alone are insufficient.

### Stage 2 — Retain experts according to useful reuse

**Hypothesis:** Activation history and miss cost can improve residency beyond
recency-only replacement without creating another large anonymous RAM cache.

- Collect bounded per-layer activation, reuse, and miss-cost statistics.
- Evaluate one replacement/admission policy at a time: for example, decayed
  frequency with recency, or reuse weighted by transfer cost and resident bytes.
- Preserve pinned readers/fills and the directory's generation/rollback rules.
- Study the transition from prompt processing to generation. Broad prompt scans
  should not erase a useful generation working set without a measured benefit.
- Keep the managed RAM cache disabled for this experiment so effects remain attributable.

**Benchmark:** First compare identical greedy work with the last accepted policy.
Report misses and transferred bytes alongside token latency. Then check a code
or tool workload to detect a policy overfitted to repeated prose.

**Exit:** Repeatable benefit with bounded metadata and no lifecycle regression,
or a documented rejection. Record actual policy overhead too.

### Stage 3 — Predict transfers with bounded confidence

**Hypothesis:** A predictor appropriate for mixed residual streams can make
useful experts ready early enough to hide part of their service latency.

- Derive candidates from valid activation history, measured co-activation, or a
  compatible predictor. Evaluate a lightweight trace-based candidate first.
- Keep the native router authoritative: prediction changes data movement only,
  never which experts contribute to the result.
- Budget prefetch bytes, distance, and occupancy using confidence and measured
  transfer/compute slack. Preserve demand priority and cancel useless queued work.
- Account for false-prefetch bandwidth and displacement. Consider a learned
  predictor only if the lightweight approach leaves a justified opportunity;
  record any additional training/checkpoint dependency separately.

**Benchmark:** Compare prefetch off/on after the Stage 2 decision. Record useful
prefetch bytes, lateness, unused bytes, displacement, and exposed wait.

**Exit:** Lower generation latency with unchanged routing and acceptable source
traffic. High prediction recall by itself is not a successful result.

### Stage 4 — Enable MTP across both GPU owners

**Hypothesis:** Multiple accepted tokens per verification round can amortize
target execution enough to outweigh drafting and expanded expert traffic.

- Extend placed execution to the existing compatible prediction head and
  verification path, including shared tables and cross-owner feature transport.
- Implement accepted-prefix commit and rejected-suffix repair across attention,
  n-gram/history, mixed residual streams, and both owners' allocation lifetimes.
- Include draft workspace/weights and the verification expert union in admission.
- Begin with a small draft budget; adapt proposal length only after a fixed-budget
  result is understood. Do not change acceptance rules to obtain a speed claim.

**Benchmark:** Compare the accepted plain build with its MTP variant. Report
time per committed token including draft, verification, and repair. Start with
canonical correctness fixtures, then measure code/tool or ordinary continuation;
copying a known passage can make acceptance unrepresentative.

**Exit:** Correct rollback/recovery and a real serving benefit after all draft
and memory costs. A compatible package or high acceptance alone is insufficient.

### Stage 5 — Optimize packed kernels and resident representations for SM86

**Hypothesis:** Exact-shape memory layout, dequantization, and kernel dispatch
can improve resident execution or fit more useful experts in VRAM.

- Profile actual resident formats and matrix shapes. Existing GEMV, grouped
  GEMM, and fused gate/up paths are the starting point.
- Try one justified change at a time: tile/dispatch choice, packed layout, or
  dequantization/load scheduling. Include speculative verification shapes if Stage 4 is retained.
- Preserve supported GGUF semantics. A different quantization scheme is a
  separate quality experiment, not an invisible kernel optimization.
- Account for any new packing artifact's size, preprocessing, transfer cost,
  cache identity, and invalidation. Avoid enlarging stored records without
  measuring the resulting offload cost.
- Respect the existing numerical qualification policy and fixed tuning budget.

**Benchmark:** Qualify affected operators, measure the relevant kernel shapes,
then confirm serving performance with the same workload and storage policy.

**Exit:** A serving or demonstrated memory-capacity improvement without precision
regression. A faster isolated kernel does not establish an end-to-end gain.

### Stage 6 — Decide whether selected misses should execute on the CPU

**Hypothesis:** For some misses, AVX2 CPU execution plus activation/result
transfers costs less than moving expert weights and executing on the GPU.

- Measure a crossover using this CPU's actual memory bandwidth and supported
  instructions. Do not extrapolate AMX/AVX-512 workstation results to the 5900X.
- Reuse compatible CPU operators where available. Keep original router choices,
  expert weighting, accumulation order/precision contracts, and failure semantics.
- Introduce bounded CPU work only for measured beneficial cases. Account for
  contention with source workers, rendering, and the operating system.
- Keep GPU execution as the control; do not globally move experts to the CPU.

**Benchmark:** Compare matched expert work and then the affected serving case.
Measure CPU queueing, activation/result traffic, GPU idle time, and memory use.

**Exit:** A demonstrated crossover and an attributable serving benefit. Defer
this stage if no useful crossover exists on the current hardware.

### Stage 7 — Adapt placement and prompt grouping to the workload

**Hypothesis:** Measured layer cost and phase-specific memory needs can select
better ranges/group sizes than a fixed equal-block split and one prompt group.

- Use observed layer work, attention/readout cost, and per-owner capacity to
  evaluate a small justified placement change before adding automatic selection.
- Evaluate prompt grouping separately. Account for prompt amortization, scratch
  memory, expert displacement, and subsequent generation latency.
- Preserve global layer identities, state ownership, gathered-table history,
  transport ordering, and per-owner memory accounting.
- Treat pipelining independent sequences as an aggregate-throughput experiment.
  A single token's layer dependencies remain sequential. Reassess topology
  before considering tensor parallelism; current peer reads are unsupported.

**Benchmark:** Separate placement and grouping A/B decisions. Report short- and
long-prompt effects. Add concurrent sequences only for a throughput claim.

**Exit:** A reproducible workload-specific configuration or measured selection
policy, with explicit tradeoffs and preserved admission/cancellation behavior.

### Stage 8 — Consolidate and check the final configuration

- Compare the last accepted implementation with the initial Stage 0 baseline
  and the independent llama.cpp reference under matched conditions.
- Check cumulative effects. Individually useful changes can contend for the
  same bandwidth, memory, or overlap opportunity when combined.
- Qualify affected code/tool behavior, retained prefixes, cancellation/recovery,
  and context capacity. Run the expensive maximum-context case only when needed.
- Preserve existing supported single-GPU behavior and use a relevant smaller
  model as a regression control when a shared path changed.
- Publish individual measurements, spread, failures, tradeoffs, reproduction
  information, and final defaults. State any remaining generation gap directly.

**Exit:** An evidence-backed final configuration and a clear disposition for
every attempted stage. Retained features, optional experiments, and deferred
research must be distinguishable in the result.

## Experiment ledger

Use one row per independent candidate. Add the hypothesis and primary metric
before running it; fill measured fields only from saved artifacts.

| Experiment | Stage | Baseline / candidate identity | Workload / run artifacts | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| Historical RAM cache comparison | Background | Recorded build/digest above | `20261018T194603Z-host-cache-comparison-8c3f34` | No reliable managed-cache win | Keep retention opt-in |
| Current baseline and attribution | 0 | `5cac0a6f`, verified executable `d5fff37c0b29…` | `20261019T063546Z-stage0-transfer-profile-92e02a`, 758 + 64 | Exact output; measured prefill 12.994 s, generation 8.509 s; native counters below | Preserve control; implement and compare Stage 1; no full GPU breakdown claimed |
| Full prompt + decode transfer batching | 1 | Control `d5fff37c0b29…`; candidate `5cac0a6f` + `f888a58f…` | Build `20261019T065412Z-stage1-build-c35edb`; short lifecycle and A/B/B/A next | Build/provenance passed; first cold request failed with `BackwardClock` | Reject prompt batching; qualify decode-only revision before timing |
| Decode-only transfer batching | 1 | `5cac0a6f` + `09325a64…`; preserved control unchanged | Build `20261019T072428Z-stage1-build-a8119e`; lifecycle `20261019T072541Z-stage1-lifecycle-72e4ea` | Lifecycle and all 12 comparison outputs passed; mean difference below control spread | Not promoted; preserve synchronous control and candidate patch |
| Decode reuse during prompt seeding | 2a | `801a56be` + `60459c8a…` | Build `20261019T075226Z-stage2a-build-75b7e4`; lifecycle `20261019T075554Z-stage2a-lifecycle-2c4074`; comparison `20261019T075917Z-stage2a-comparison-b3e32b`; tools `20261019T081041Z-stage2a-tools-99971b` | All lifecycle and semantic gates passed; generation time −26.585%, logical generation reads −46.570%; prefill +28.945% | Retain optional decode-reuse profile for repeated requests; preserve original default |
| Reuse-weighted decode victims | 2b | `c9cbfcbf` + `07b58952…`; control is frozen Stage 2a | Build `20261019T082201Z-stage2b-build-5a6f9f`; lifecycle `20261019T082536Z-stage2b-lifecycle-eed91a`; comparison `20261019T082747Z-stage2b-comparison-baab09`; tools `20261019T083547Z-stage2b-tools-4a56cc` | Lifecycle, all 12 comparison outputs and held-out tools passed; generation −16.496%, logical generation reads −30.102%, prefill +0.219%, total −6.319% | Retain optional replacement alongside Stage 2a seeds; preserve prompt/LRU defaults |
| Bounded repeated-miss prefetch | 3 | `d2b3b935` + `67eae120…`; frozen Stage 2b control | Build `20261019T085958Z-stage3-build-a9e8ac`; lifecycle/admission `20261019T090124Z-stage3-lifecycle-562553` | All six checks passed; 6,921 repeat trials, zero repeated misses or issued bytes, including the third independent request | Reject this inactive heuristic; no timing claim or source promotion; defer differently conditioned prediction |


### Stage 0 progress — baseline and diagnostic correction

The uninstrumented warmup and measured request both returned the exact expected
308-character prefix, with 758 prompt and 64 generated tokens and no prefix reuse.
The measured request reported 12.994019 s prefill and 8.508615 s generation
(7.521789 tokens/s); warmup reported 32.607956 s and 10.574497 s respectively.
This one measured request establishes a reproduction point, not a speedup or
noise estimate. Its process I/O delta was 46,295,341,696 logical read bytes and
12,355,145,728 physical read bytes across the complete request. This is not a
decode-only disk measurement and does not prove that all source bytes came from SSD.

Artifacts are under `inference/evaluation/results/20261019T063546Z-stage0-transfer-profile-92e02a/`:
`build.json`, `provenance.json`, `baseline-observation.json`, I/O snapshots,
rendered fixtures, options, allocation samples, and copied reproduction scripts.
The engine SHA-256 is
`d5fff37c0b29392aaf66e5700f5a85d8aa1cf2a62b85fc444ca6b2b4dba721a9`.
The existing checkout's executable provenance verified without rebuilding code.

The profiled response also passed exact-output validation, but its wrapper did
not finish within the 90-second cleanup limit. The run correctly records failure.
Nsight subsequently generated its report. Inspection found no decode kernels in
that report. Missing shutdown flush was an initial hypothesis, not an established cause.
Do not use that incomplete report to attribute decode or the slower instrumented
prefill as an engine regression. Repeat only the diagnostic with CUDA tracing,
periodic trace flush, and graceful SIGINT; omit OS-runtime tracing to reduce
instrumentation overhead. No implementation or tuning changes accompany this retry.

The CUDA-only retry `20261019T064430Z-stage0-cuda-profile-1446a5` completed
successfully, including graceful shutdown, and produced another exact 64-token
answer. It still captured only startup kernels, despite periodic flushing and
node-level graph tracing. This disproves shutdown flushing as a sufficient
explanation. Its I/O snapshots measured 10,671,763,456 physical bytes before the
first delivered token and 606,785,536 afterward; corresponding logical reads
were 47,870,617,600 and 11,474,049,920 bytes. These are client-delivery boundaries,
which include possible lookahead, rather than exact executor phase boundaries.
One final bounded diagnostic enables graph-level/all-API tracing and existing
expert counters. A complete GPU arithmetic/wait breakdown remains unmeasured;
no incomplete trace authorizes such a breakdown. Capture settings follow the
[NVIDIA Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

### Stage 1 first qualification — failed, preserved for diagnosis

`20261019T065835Z-stage1-lifecycle-b7732a` failed its first cold request before
valid output. The expert watcher panicked in scratch-directory admission with
`BackwardClock`; poisoned-state cleanup then aborted the worker. The HTTP client
reported an incomplete chunked response. Build/provenance success did not qualify
this candidate. Preserve the log, failed observation, source identity and initial
candidate patch; do not benchmark or promote it. The reduced 758 + 16 reproduction
`20261019T070532Z-stage1-clock-repro-a9f421` also failed: scratch sequence 3 at
local layer 0 was requested after the directory had already observed sequence 3
at layer 2, both in group 2. Progress agreed with that later observed clock.
The bounded event trace in `20261019T071145Z-stage1-clock-repro-0103ef`
shows failure words already set: scratch sequence 2 at layer 0 was followed by
sequence 3 at layers 1 and 2 with failure=3, then the stale layer-0 request.
Thus the backward-clock panic is secondary to an earlier scratch failure. Expose
that first stored failure before changing clock semantics. Do not suppress or
relax the clock invariant to admit the candidate. The control source/executable remain unchanged.

### Stage 1 revision — restore prompt copies, isolate generation batching

The final diagnostic `20261019T071713Z-stage1-clock-repro-e6d0c0` preserves
`failed-candidate.patch` and the first stored failures. Layer 0, scratch round 2
waited for sequence 2 at arrival 1 but observed 3. The next stored error was CUDA
illegal memory access (700) during a later source upload; the clock panic followed.
The exact cause of that prompt protocol violation remains unresolved. Do not
label the clock panic as the original fault or claim it is fixed.

Reject asynchronous batching in the prompt scratch path for this candidate.
Restore its original per-tensor synchronous copies exactly and narrow the next
experiment to decode transfers only. This is a revised experiment, not a repair
claim for prompt batching. Keep the failed patch/artifacts available for later
protocol work. The owned completion API and ready-expert decode queue remain;
qualification and alternating timing must pass before either is promoted.

### Stage 1 decode-only qualification and comparison declaration

The revision with the original prompt scratch copy loop restored has source
identity `5cac0a6f` + dirty digest
`09325a64dae075188d80d22426dbb4edbc52c61ef553fe6f2a29ec84fd054e07`.
Its release build and both candidate/control executable proofs passed in
`20261019T072428Z-stage1-build-a8119e`. The source patch is saved with that run.

Lifecycle `20261019T072541Z-stage1-lifecycle-72e4ea` passed cold, retained,
prefix-hit, cancelled-stream and recovery checks. Completed answers were the
same exact 308-character / 64-token prefix. The hit and recovery reported 757
cached prompt tokens; cancellation closed after four content events (15 exact
characters) before terminal delivery, and recovery remained healthy. These are
correctness observations, not controlled performance results.

Next comparison: unchanged 758-input + 64-output fixture, plain decoding,
lookahead on, prompt group 8, configured context 1,024, placement 24 + 24,
managed cache disabled, and prefix reuse disabled on every request. Use two
fixed warmups before one measured request per launch, in A/B/B/A order, under
one canonical lock and thermal gate. The extra warmup is justified by Stage 0's
large run-order/cache effects. Compare identical rendered tokens/templates and
greedy output. Start with two measured requests per side and apply the declared
3%/noise and secondary-regression bars before any promotion.

### Stage 1 measured decision — keep the synchronous control

**Verdict: no promotion.** The decode-only revision passed correctness, but its
speed benefit was not established beyond the observed variation. Do not present
the mean difference below as a repeatable improvement.

Controlled comparison: `20261019T072922Z-stage1-comparison-ab56d2`. Canonical lock
and thermal gate passed; all eight warmups and four measured responses were
identical exact prefixes with 758 input and 64 output tokens. Both harnesses
rendered identical tokens and template identities. No source/executable changes
occurred during the run. `comparison.json`, `observations.jsonl`,
`timing-summary.json`, `decision.json`, both executable identities, I/O snapshots,
allocation samples, and reproduction scripts are saved with the run.

| Order / condition | Measured prefill | Measured generation | Model total | Physical reads after first token |
| --- | ---: | ---: | ---: | ---: |
| A / original control | 12.742 s | 6.308 s | 19.051 s | 1,191,325,696 B |
| B / decode-only events | 5.072 s | 4.648 s | 9.720 s | 21,671,936 B |
| B / decode-only events | 9.522 s | 4.785 s | 14.307 s | 106,840,064 B |
| A / original control | 4.906 s | 4.564 s | 9.470 s | 21,671,936 B |

Mean generation was 5.436336 s for A and 4.716696 s for B, an apparent 13.2376%
reduction. The 0.719640 s difference is below A's 1.744034 s spread; B's spread
was 0.136865 s. The last control was faster than both candidate measurements,
including the candidate with the same 21,671,936-byte physical-read delta.
Substantial OS-cache/run-order effects remain despite two fixed warmups. These
observations do not establish either a repeatable gain or a general regression.
No extra comparison is needed to reject default promotion under the declared bar.

Logical reads after first token were approximately 12.963 GB in every measured
condition. Moving these bytes through file-cache service and H2D remains work
even when physical disk traffic is small. Delivery-window counters include
possible lookahead. Across the four launches, minimum sampled device headroom
was 2,825,519,104 B on owner 0 and 2,297,167,872 B on owner 1. Minimum host
headroom was 57,681,203,200 B; process RSS peaks were 7.319–7.344 GB. No memory
pressure refusal was observed in this bounded comparison.

Keep the original source and executable as the accepted baseline. The full
prompt/decode patch remains rejected for correctness. The decode-only patch,
which qualified but did not meet the performance bar, remains an unmerged
experiment in its isolated worktree and in the saved build artifact. Neither
implementation is promoted to the working branch. No new default or runtime
knob is introduced by this stage's disposition.

### Stage 2a next experiment — preserve useful decode experts at prompt seeding

**Hypothesis:** The transition from broad prompt routing to generation replaces
useful decode residency. Recording bounded decode reuse and using it when the
prompt seeds its last scratch member can reduce subsequent expert movement.
This targets source bytes rather than another transfer-completion refinement.

- Start from the accepted synchronous-transfer implementation.
- First change prompt seeding only. Keep ordinary arena victim selection as LRU;
  evaluate a frequency-based replacement rule separately after this decision.
- Count observed decode uses from GPU slot stamps, including all-hit steps.
  Observation only at host misses would bias the history. Deduplicate repeated
  observations of the same slot/step, retain ownership/generation checks, and
  bound metadata by admitted slot/expert counts. Report its overhead.
- Keep prompt scans from increasing decode-use counts. Preserve actual GPU use,
  conversion commits, active readers/fills and accepted state semantics. Cache
  history is a placement hint, never permission to read uncommitted weights.
- At seeding, consider resident experts with useful decode history as well as
  the prompt's routed candidates. A retained expert outside the prompt union
  needs no new source read. With no history, preserve the original prompt rank.
- Use a bounded decay so old topics can lose priority. Declare one initial decay
  rule; avoid a parameter sweep before measuring whether the mechanism helps.
- Primary metric remains uninstrumented generation latency, with logical source
  bytes and misses as mechanism evidence. Qualify lifecycle first, use the same
  short A/B/B/A case, and broaden to code/tool behavior only after a useful result.

### Stage 0 decision — use owned completion per ready transfer batch

Final diagnostic: `20261019T064804Z-stage0-graph-profile-5d6cd6`, with exact output,
no MTP, no prefix reuse, no managed cache, group 8 and the same placement. It ran
without a thermal gate because this is attribution, not a performance comparison.
Its saved `attribution.json` and `logs/magnitude-profile.log` record:

- Across both owners' 63 completed decode sequences after the prompt: 3,652
  expert misses, 11,389,337,600 demand bytes, no speculative source bytes, two
  revived records, and 2,989.599 ms total watcher service.
- Watcher service includes file waiting, transfer completion, polling and
  publication. It is not measured H2D time. Its sum cannot be added to kernel
  durations or treated as a non-overlapping wall-time fraction.
- Between first delivered token and terminal response, physical reads increased
  by 301,928,448 bytes and logical reads by 11,470,969,856 bytes. Most reads in
  this observed generation window were served without physical disk I/O.
  Delivery boundaries and lookahead prevent interpreting this as an exact
  executor phase counter.
- Instrumented model timings were 16.616 s prefill and 5.057 s generation. The
  large difference from the single earlier untraced request demonstrates cache
  and run-order effects. It does not show a speedup from profiling.
- All-API/graph-level tracing still omitted production graph launches and
  kernels. The cause remains unresolved. Retain the raw traces, avoid a fourth
  profiler retry, and make no arithmetic/H2D/coordination percentage claim.

The baseline and existing counters justify the first bounded implementation:
queue the three ready tensor copies for an expert before waiting; let other
already-ready experts enqueue while those copies run; publish each expert only
on successful event completion. The initial scratch-round batching attempt was
subsequently rejected; its original synchronous path is restored in the qualified revision. Keep file/expert policies and kernels fixed.
A completion owns both source and destination allocations, and dropping one
must finish its queued copies before either staging or device slots can be reused.
The comparison's primary metric is uninstrumented generation time; the 3% and
noise bars above apply. Preserve the control source/executables in the main
checkout while developing the candidate in an isolated worktree.

For each completed row record:

- Exact change and configuration; build and rendered-input identities.
- Predicted mechanism, primary metric, and declared promotion bar.
- Individual timings and spread; correctness, lifecycle, and memory outcomes.
- Relevant cache/transfer/speculation counters and instrumentation conditions.
- Retain/revise/revert/defer decision, accepted commit if any, and next action.

## Research sources

These sources inform mechanisms. Their published speedups use other workloads
and hardware and are not forecasts for this system.

1. [vLLM offload configuration](https://docs.vllm.ai/en/latest/api/vllm/config/offload/)
   — asynchronous H2D prefetch and the memory/latency tradeoff of prefetch distance.
2. [MoE-Infinity](https://arxiv.org/abs/2401.14361)
   — activation-trace-informed expert replacement and prefetch.
3. [SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
   — MTP, verification budgets, and method-specific constraints.
4. [Fiddler](https://arxiv.org/abs/2402.07033)
   — CPU/GPU orchestration to reduce costly expert-weight movement.
5. [KTransformers](https://github.com/kvcache-ai/ktransformers) and its
   [AMX discussion](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)
   — heterogeneous execution and the importance of CPU-specific capabilities.
6. [Marlin](https://github.com/IST-DASLab/marlin)
   — Ampere-compatible packed-weight layouts, dequantization, and memory scheduling.
7. [llama.cpp server configuration](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
   — explicit CPU expert placement and supported multi-GPU split modes.
8. [FlashInfer GEMM interfaces](https://docs.flashinfer.ai/api/gemm.html)
   — grouped execution interfaces and architecture-specific backend selection.

### Stage 2a implementation declaration

Candidate `reuse-seeding` starts from the accepted branch and changes prompt
seeding only. The experimental `MAGNITUDE_EXPERT_SEED_POLICY=decode-reuse`
selects decayed observed decode-step counts before prompt frequency; unset or
`prompt` preserves the existing ranking. Counts halve every 64 bank sequences.
Each admitted (layer, expert) has one 8-byte counter. Active per-layer row-class
metadata expires at completion, including unfenced abandonment. Readable bank
identities are observed once per group at progress and physical completion, so
all-hit uses are included; repeated observations of one expert/step deduplicate.
Scratch scans do not increment these counts. No weight payload cache, CUDA kernel,
transfer protocol or LRU victim rule changes. Qualification and performance
results follow below. Stale debug build output was cleaned before the release build;
the verified release control is preserved.

### Stage 2a qualification — passed

Release build `20261019T075226Z-stage2a-build-75b7e4` verified both candidate and
control executables. Candidate dirty digest is
`60459c8aebe9e84585b6247f5d77f8628d82b96b1b98287e3af6327efd65513f`.
The accepted control source and binary are unchanged. The build cleaned 8 GB of
stale debug artifacts while retaining the release control.

Lifecycle `20261019T075554Z-stage2a-lifecycle-2c4074`, with decode-reuse selected,
passed all five checks. Completed responses were the same exact 308-character /
64-token prefix; hit and recovery reused 757 prompt tokens. Cancellation closed
after four content events (15 exact characters) before terminal delivery, and
recovery was healthy. The log confirms 98,304 bytes of expert counters per owner,
with existing expert source/streaming diagnostics enabled. These timings are not
performance comparisons.

The next run keeps the Stage 1 comparison controls: two fixed warmups per launch,
A/B/B/A, 758 + 64 tokens, context 1,024, plain/lookahead/group-8, 24 + 24 placement,
managed cache off, prefix reuse off and tracing off. A selects the original prompt
policy/control binary; B selects decode-reuse in the candidate. Identical rendered
tokens, template and generated work remain required. Promotion still needs at
least 3% generation improvement beyond observed spread and a useful source-byte
reduction; a positive result must then pass code/tool coverage.

### Stage 2a controlled result — generation benefit with a prefill tradeoff

`20261019T075917Z-stage2a-comparison-b3e32b` completed under one lock and a
passing thermal gate, with all 12 responses exact and identical rendered tokens.

| Condition / order | Prefill | Generation | Model total |
| --- | ---: | ---: | ---: |
| A / original prompt seeds | 5.396 s | 4.574 s | 9.969 s |
| B / decode reuse | 8.833 s | 3.578 s | 12.411 s |
| B / decode reuse | 4.943 s | 3.154 s | 8.097 s |
| A / original prompt seeds | 5.288 s | 4.596 s | 9.884 s |

Mean generation fell from 4.584820 s to 3.365956 s (26.585% lower time).
The 1.218864 s difference exceeds both observed spreads: 0.022477 s for A and
0.424655 s for B. Mean logical reads after first token fell from about 12.972 GB to
6.931 GB (about 46.6%); this supports the intended reduction in expert movement.
Delivery windows still include possible lookahead. Both harnesses rendered the
same chat template and tokens, and all requests generated the same 64-token text.

Mean prefill increased from 5.341716 s to 6.887876 s (28.9%); mean model total
increased from 9.926536 s to 10.253832 s (3.3%). The slower B prefill also had
more physical reads (4.441 GB, versus 0.324 GB on the other B pass), but the
observations do not establish a generally safe prefill default. Keep the original
policy as default. Artifacts include raw timings, I/O windows, allocations,
identities and reproduction scripts; `decision.json` records the final decision.

### Stage 2a held-out gate and disposition

`20261019T081041Z-stage2a-tools-99971b` passed for both original and candidate
builds. Each engine first generated the exact 64-token prose prefix, then served
the canonical `tools/bfcl/simple_python_0` fixture in the same process with prefix
reuse disabled. Both rendered identical 366-token tool requests and generated
53 tokens ending in the expected `calculate_triangle_area` call with exactly
`base=10`, `height=5`, `unit="units"`. The canonical validator matched the full
call and arguments. This is a bounded semantic/topic-transition gate, not a
tool-speed or distributional-quality measurement.

Peak process RSS across the controlled comparison was 6.841–6.846 GiB for A
and 6.817–6.818 GiB for B. Classified host allocations were unchanged at
2,725,803,264 bytes; additional Rust policy counters are separately documented
above. Observed host availability stayed above 57.8 GB, including reclaimable
cache. Minimum device headroom was 2,825,519,104 and 2,297,167,872 bytes;
no reserve or allocation failure occurred. RSS excludes additional Linux file
cache, and the similar RSS does not establish identical total memory traffic.

**Retain the implementation as an optional profile**, enabled with
`MAGNITUDE_EXPERT_SEED_POLICY=decode-reuse`. Unset or `prompt` keeps the original
default. Its measured benefit applies to this repeated-prose workload after two
warmups per process. No cold-request, tool/code speed, single-GPU or llama.cpp
win is established. The measured prefill regression exceeds the secondary
acceptance bar, so this evidence does not justify a default change.

### Stage 2b declaration — reuse-weighted decode victims

Start from retained Stage 2a source `c9cbfcbf` in a separate `reuse-replacement`
worktree. The control remains the frozen Stage 2a executable, selecting
`MAGNITUDE_EXPERT_SEED_POLICY=decode-reuse` on both sides. The candidate adds
`MAGNITUDE_EXPERT_REPLACEMENT_POLICY=decode-reuse`; unset or `lru` preserves
recency-based replacement. The two policy selections remain independent.

Only small-row demand placement changes its victim ranking. Empty slots remain
first; among readable, unleased slots in the existing compatible arena, evict
the lowest **decayed decode-use count × stored gate/up/down bytes**. Existing
recency, layer and slot order break ties. Use the existing 64-sequence count
decay and bounded expert history. Stored bytes are a static reload-cost proxy,
not measured SSD time; compatible arena records have the same resident geometry,
so no per-victim resident-size normalization is required. Prompt scratch seed
placement keeps its existing ranking and quota. Do not change eligible victims,
read/fill leases, conversion publication, rollback, transfers or source priority.

Fully build the candidate, then run the short cold/retain/hit/cancel/recovery
fixture gate. If it passes, use the existing 758 + 64, two-warmup A/B/B/A with
managed host cache disabled and tracing off under one lock and thermal gate.
Require exact matching rendered/generated work and at least 3% generation benefit
beyond each side's observed spread. Compare logical and physical reads and memory
as well. A promising result needs the held-out tool transition gate; a noisy or
negative result leaves this policy unpromoted. Stage 2a's win is not evidence for
this separate mechanism.

Release build `20261019T082201Z-stage2b-build-5a6f9f` completed in 3m 07s and
verified both candidate and frozen Stage 2a engine/harness executables. Candidate
identity is `c9cbfcbf` plus
`07b5895280587a26eb178bd8bb8f8f10a3dc011417d9b19dfc8997c2e1d5c7a4`.
The candidate adds no weight cache or new per-expert counters; seed and victim
policies share the existing history. An absent/`lru` replacement policy preserves
the original victim path. The complete patch and build launcher are saved with
the result. Lifecycle qualification follows with both reuse policies enabled.

### Stage 2b lifecycle qualification — passed

`20261019T082536Z-stage2b-lifecycle-eed91a` passed cold, retain, hit, cancellation
and recovery checks with both reuse policies enabled. All four completed
responses were the exact same 308-character / 64-token prefix. Hit and recovery
reused 757 prompt tokens. Cancellation closed after four content events / 15
exact characters before terminal delivery; the following recovery stayed healthy.
Both owners reported the same 98,304-byte counter inventory as Stage 2a.

Instrumented diagnostics across 63 decode sequences / two banks found 3,565
misses and 11,135,283,200 demand bytes after the cold request's scratch steps;
the corresponding retain window had 1,734 misses and 5,422,387,200 demand bytes.
Both had zero prefetch bytes. These are priming observations from one candidate
process, not matched A/B speed or miss reductions. Timed A/B/B/A follows with
tracing disabled and the frozen Stage 2a profile as control.

### Stage 2b controlled result — lower generation time and traffic

`20261019T082747Z-stage2b-comparison-baab09` completed under one machine lock
and a passing thermal gate. Both sides selected reuse-based prompt seeds; A
kept LRU demand victims and B selected reuse-weighted demand victims. All 12
responses were the same exact 64-token prefix; rendered tokens and template
identities matched, with 758 input tokens and no retained-prefix reuse.

| Condition / order | Prefill | Generation | Model total |
| --- | ---: | ---: | ---: |
| A / Stage 2a profile | 4.955 s | 3.141 s | 8.096 s |
| B / reuse-weighted victims | 4.926 s | 2.623 s | 7.549 s |
| B / reuse-weighted victims | 4.904 s | 2.639 s | 7.543 s |
| A / Stage 2a profile | 4.854 s | 3.161 s | 8.014 s |

Mean generation time fell from 3.150704 s to 2.630979 s, **16.496% lower**.
The 0.519725 s improvement exceeds A's 0.019609 s and B's 0.016191 s observed
spreads, passing the declared 3%/noise bar. Mean prefill changed from
4.904444 s to 4.915186 s (+0.219%, within observed variation); mean model total
fell from 8.055148 s to 7.546165 s (−6.319%). No secondary timing bar failed.

Mean logical reads after first-token delivery fell from 6.927811 GB to
4.842394 GB (−30.102%). Physical reads in these windows were only 3.072–6.144 MB
for A and 6.144–6.148 MB for B. This supports reduced expert movement while
showing that an SSD-byte reduction does not explain the win. Delivery windows
can include lookahead. Peak process RSS was 6.816–6.847 GiB across all launches;
classified host/device allocations were unchanged, and no reserve failure
occurred. The result contains complete memory observations and provenance.

The performance evidence is limited to the repeated-prose workload after two
fixed warmups per launch, compared with Stage 2a in the same session. Do not
multiply improvements from different sessions into a combined speedup. Preserve
both existing defaults. `analysis.json` records calculated timings, I/O and
memory; `decision.json` records the final disposition after the semantic gate.

### Stage 2b held-out gate and disposition

`20261019T083547Z-stage2b-tools-4a56cc` passed both builds through prose priming
and the canonical BFCL tool fixture in the same process. Both tool requests
rendered to 366 tokens and produced the expected 53-token
`calculate_triangle_area(base=10, height=5, unit="units")` call. The canonical
validator matched the complete function and arguments. Both prose prefixes were
also exact. This is a semantic/topic-transition check, not tool-speed evidence.

**Retain Stage 2b as an optional extension to the measured Stage 2a profile:**

```sh
export MAGNITUDE_EXPERT_SEED_POLICY=decode-reuse
export MAGNITUDE_EXPERT_REPLACEMENT_POLICY=decode-reuse
export MAGNITUDE_HOST_EXPERT_CACHE_BYTES=0
```

Unset policies still select prompt-frequency seeds and LRU victims. The new
replacement setting alone is not the measured profile. This result establishes
an additional benefit against Stage 2a on the repeated-prose fixture; broad
workload, cold-request, single-GPU, combined-original-baseline and llama.cpp
claims remain unestablished. The source/design patch is retained, and both
candidate and control worktrees/executables remain frozen for the next stage.
Stage 2 evaluation is complete: useful prompt seeding and demand replacement are retained
as explicit options, without a new RAM payload cache or changed CUDA kernels.

### Stage 3 declaration — bounded repeated-miss prefetch

Start from retained Stage 2b source `d2b3b935` in an isolated `history-prefetch`
worktree. Both sides keep the Stage 2 seed/replacement profile and managed host
cache disabled. The candidate adds `MAGNITUDE_EXPERT_PREFETCH_POLICY=history-repeat`;
unset or `native` preserves the existing predictor behavior. Mixed-residual
future-router prediction remains disabled.

Learn each expert's **device miss at step q given its device miss at q−1** from
actual directory resolutions. A completed small-row step with no demand request
for a layer supplies negative evidence; only adjacent, valid, completed small-row
steps train. Scratch steps and failed/abandoned work cannot train this statistic.
Counters halve every 64 bank sequences. Require at least four decayed trials and
at least three-quarters successful repeats. This is an empirical admission
heuristic, not a statistical confidence interval. At most two eligible experts
per target layer are ranked by repeat fraction and supporting trials.

At step submission, consider the first two local layers; after serving a demand,
consider the next two local layers while the GPU resumes. Predictions come only
from the preceding step's actual misses in that target layer. Skip GPU residents,
intact free staging copies and already-staged records. Limit this history path to
four outstanding speculative records and 16 MiB of stored payload per owner, within the
existing hint shelf. It reserves no GPU slots and moves no bytes to device memory
until the real router demands them. Existing demand priority, source-ticket
lifetimes, expiry and cancellation remain authoritative.

Record admitted payload/source bytes, ready versus late demand matches, unused
bytes at expiry/reclamation, cancellation, forced staging reclamation and unresolved
records. These counters describe requested payload, not physical SSD reads;
process I/O remains the physical/logical cross-check. A running read retains its
ticket through completion and may still match later demand. The predictor adds bounded
per-expert counters and one miss inventory per in-flight step plus its immediate
predecessor, without a weight cache. Its counters retain physical cache effects
without changing accepted numerical state.

Build fully before the lifecycle gate. Include a third independent prose request
to exercise the same two-warmup training boundary as the timed fixture, followed
by prefix reuse, cancellation and recovery. If it admits useful work, compare
untraced A/B/B/A against frozen Stage 2b, requiring the usual 3%/noise bar and
bounded traffic. Zero admission at that boundary rejects this predictor for the
fixture without spending a full timing comparison on an inactive mechanism.
A promising result then needs the held-out tool transition gate. Preserve any
failure and state the measured limitation before moving to Stage 4.

Initial build `20261019T085601Z-stage3-build-4f272a` passed. Review then found
that counter snapshots also need all-hit completions, so an incremental build
`20261019T085958Z-stage3-build-a9e8ac` includes that diagnostic correction and
verified both sides' engine/harness provenance. No model validation ran on the
earlier build. The final candidate identity is `d2b3b935` plus
`67eae1200f40849471a7390d5465ad37fe17438d40a2647612029c307c39318e`.
The complete patch includes the new predictor module. All source remains frozen
through the six-check lifecycle/admission probe, with streaming diagnostics enabled.

### Stage 3 result — no repeated device-miss opportunity

`20261019T090124Z-stage3-lifecycle-562553` passed all six checks. Cold, warm,
retain, hit and recovery each produced the exact same 308-character / 64-token
prefix. Hit and recovery reused 757 prompt tokens. Cancellation closed after
four content events / 15 exact characters before terminal delivery and recovery
remained healthy. Both owners reported 147,456 bytes of predictor counters,
in addition to the unchanged Stage 2 counters.

The 678 cumulative snapshots covered 339 physical bank sequences per owner,
including all-hit completions. The predictor observed **6,921 adjacent valid
small-row device-miss trials**: 3,757 on one owner and 3,164 on the other.
There were **zero repeated device misses**, zero confident candidates and zero
issued prefetch records/bytes. The third independent request's 63 decode-step
window alone supplied 1,644 trials, again with zero repeats or admissions.
The ready, late, unused, cancelled, unresolved and forced-reclamation counters
were consequently all zero. Recorded payload conservation and budget checks
passed, but these zero counters do not qualify an active prefetch lifetime path.

The same expert needing a new device load on the immediately following step
did not occur in this fixture. This finding does not say that expert activations
never repeat: reuse served by the GPU cache is not a device miss. The chosen
miss-recurrence event gives this predictor no observed opportunity after Stage 2
retention. Planning accounted for 1.222190 ms across the complete instrumented
gate; that excludes learning and other host work and is not an A/B latency result.

**Reject this candidate for the tested profile and preserve Stage 2b.** The
declared third-request admission gate was inactive, so a full timing comparison
and held-out tool gate were not run. No speedup, slowdown, useful-prefetch ratio
or active-prefetch cancellation qualification is claimed. The complete source
patch, both build records, raw checks, `admission-summary.json`, `decision.json`
and reproduction scripts remain available locally. Experimental source stays in
the isolated worktree and is not promoted.

Predicting newly missing experts needs different conditioning: for example,
validated cross-layer observations or a trained predictor appropriate for the
mixed residual form. That is deferred, not established by this experiment.
The existing future-router rejection remains in place. Revisit this path with
evidence of predictable new misses and exposed source wait; lowering a threshold
cannot create positive repeats in this dataset. No trained checkpoint or
additional activation-transfer path is introduced by this stage.

## Immediate next action

Begin **Stage 4: MTP across the two placed owners**, using the frozen Stage 2b
profile as the accepted plain control. Inspect prediction-head admission,
feature/table ownership, verification state and rejected-suffix repair before
changing the plain-only startup guard. Preserve the MTP proposal/acceptance and
physical lifetime contracts. Fully build and qualify real proposals, accepted
prefixes and cancellation/recovery before timing. Then work through Stages 5–8.
Stage 3's repeated-miss candidate is rejected; a differently conditioned predictor
remains deferred. Full CUDA graph attribution is still unavailable.
