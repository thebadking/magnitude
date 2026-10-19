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

A running observation at **2026-10-19 10:34:27 UTC**, during the untimed
four-position main-engine probe, found **55.886 GiB `Cached`, 0.454 GiB actually
free, and 54.013 GiB `MemAvailable`**. The harness RSS was about 6.48 GiB and
its `VmSwap` was zero. This is a single sample, not peak host occupancy; Linux
`Cached` also includes non-model files. Its process I/O showed 9.24 GB of
cumulative physical reads, including loading/setup; that counter must not be
presented as a warm-generation window. The earlier controlled warm
measurements remain the evidence for low physical generation traffic. Cache
warmth and phase must accompany any claim that disk is the bottleneck.

Snapshot and read-only observer:
`20261019T103247Z-stage4-main-features-3ebbf3/ram-103427.json` and
`ram-observer.py` under the local results directory.

A further read-only snapshot at **2026-10-19 11:52:44 UTC** found **53.363 GiB
`Cached`, 7.590 GiB actually free and 60.773 GiB `MemAvailable`** out of
62.696 GiB total. No Magnitude process was running and both GPUs were idle;
an existing sleeping llama-server had approximately 3 MiB RSS and 3.25 GiB
swapped out. System swap occupancy is not evidence that the earlier active
Magnitude probe was swapping; its sampled `VmSwap` was zero. The snapshot did
not inventory model-page residency, and `Cached` includes other files.

The practical answer is conditional: retain more useful expert pages when
physical misses are costly, while preserving runtime headroom. Here most spare
RAM is already serving the filesystem cache. Stage 2b's two candidate generation
delivery windows each had approximately **4.84 GB logical reads and 6.14 MB
physical reads**. Those windows do not show a warm disk-bandwidth bottleneck or
establish which remaining operation dominates. The existing managed-cache
comparison also failed to demonstrate a speedup. Prioritize measuring source
service time, RAM copies, uploads, conversions and GPU expert reuse before
allocating another cache. Cold loading and broader or changing expert demand
may have a different miss rate and require their own measurement.

Snapshot and its reproduction script:
`20261019T115244Z-idle-ram-diagnostic/` under the local results directory.

### Confirmed code observations

- [CUDA upload](inference/seismic/backends/cuda/src/driver.rs) calls
  `cuMemcpyHtoDAsync` and then synchronizes the transfer stream before returning.
  The [expert streaming runtime](inference/engine/executor/src/streaming/runtime.rs)
  invokes this path separately for each fetched expert tensor.
- In the frozen plain control, [placed startup](inference/engine/src/execution/placed.rs)
  requires two ranges and plain text execution. Prediction-head support in
  [the Qwen4 family](inference/engine/families/qwen4exp/src/head.rs) did not by
  itself establish MTP across owners. Stage 4 adds that capability on the
  experimental branch described below.
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
| 2 | Trace-informed GPU expert retention | 2a and 2b retained as opt-in; controlled timing and tool gates passed | 1 decision |
| 3 | Confidence-limited expert prefetch | Repeated-miss candidate rejected after no actionable predictions; other conditioning deferred | 2 decision |
| 4 | MTP across the two owners | Full-acceptance lifecycle passed; ordinary parity failed; target batching differences reproduced on plain main; diagnosis before timing | 3 decision |
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
| Placed MTP with one CUDA affine decode choice | 4 | Candidate `47f852f5` + `da537eff…`; frozen Stage 2b plain control | Build `20261019T122031Z-stage4-build-87d1df`; generated prefix `20261019T122157Z-stage4-shared-prefix-f881f0`; serving `20261019T122452Z-stage4-shared-strict-heldout-c073bf`; lifecycle `20261019T123227Z-stage4-shared-lifecycle-8d6b29`; comparison `20261019T123708Z-stage4-comparison-626259` | All correctness gates and 24 benchmark responses exact. Repeat generation −21.911%, total −0.268%; ordinary generation −2.940%, total +11.129% | Keep experimental capability; do not promote the fixed-budget profile as a general speedup. Continue Stage 5 with the plain control |


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

### Stage 4 initial implementation and lifecycle qualification

The isolated candidate now admits the catalog's own-history causal head on the
first owner. It includes head weights/state/workspace, independent conditioning
transfer slots derived from sealed head inputs, and raw F32 feature transport
from the final target owner. Independently retained prompt boundaries live on
the head owner. Target verification and state maintenance continue to use the
existing coordinated transaction and every physical owner's completion tickets.
The accepted main branch still uses its previous implementation while this
candidate completes qualification.

Build `20261019T093842Z-stage4-build-893593` completed with verified engine and
harness provenance: candidate base `47f852f5`, dirty digest
`a3dc95c59a35d94f72848accb88032fa824ff3d2ed1fa2c4bbea2ff2804f1454`.
Earlier build/readiness attempts are retained: the initial compile found two API
mismatches, and two startup attempts exposed planning/qualification paths that
mistook local head ownership for the target's raw-residual export requirement.
The correction carries that export in the target program plan and uses one
conditioning-element declaration for kernel formation and its exact-copy
qualification. No precision tolerance or proposal-acceptance rule was relaxed.

`20261019T093945Z-stage4-lifecycle-1dcc70` passed the canonical 758-input-token,
64-output-token repeated-prose checks with one generated/verified proposal per
round, the default KV codec, both reuse policies selected, and managed host
retention disabled:

- Cold, retaining, cache-hit and post-cancellation recovery responses were the
  same exact 308-character prefix, each with **31 proposals and 31 accepted**.
- Cold and retaining requests had no reused prefix; hit and recovery reused
  757 input tokens.
- The cancelled stream ended after four content events/15 matching characters,
  before its terminal response; the following recovery completed correctly.

These four completed requests exercised 124 accepted proposals and **no rejected
proposals**. They establish actual placed drafting and this full-acceptance
lifecycle; they do not qualify rejected-suffix repair or ordinary continuation.
The run was traced and untimed for promotion purposes, so no performance claim
is made. Qualify an ordinary continuation and canonical tool fixture against
the frozen accepted plain build, requiring exact greedy parity and observed
rejections, before a controlled timing comparison. Stale original build output
was cleaned to recover about 32 GiB; the accepted Stage 2b executable was kept.

### Stage 4 held-out gate — ordinary continuation diverges

`20261019T094358Z-stage4-heldout-a51b91` compared the frozen Stage 2b plain
profile with the same built MTP candidate, at context 1,024 and one fixed
proposal. Rendered tokens/templates matched. Both engines passed each fixture's
sanity gate, repeated the same 64-token passage, and made the same independently
validated BFCL triangle-area call with identical function/argument text.

The ordinary continuation **failed exact greedy parity**: plain ended naturally
after 60 output tokens; MTP generated 64 different tokens. MTP recorded 36
proposals, 26 accepted and 10 rejected on that request. The tool request accepted
all 26 proposals and completed the same 53 output tokens as plain. Request-local
tool-call IDs were excluded from comparison; names and argument text were kept.

The copying and tool results do not qualify the failed continuation. Hold timing
and promotion. First run the same canonical continuation with MTP's generated
and verified budgets both zero, retaining raw residual export and head priming,
to distinguish that path from multi-row verification/rejected-suffix repair.
The divergence's cause is not yet established; do not label it a repair defect
or ordinary floating variation without discriminating evidence.

`20261019T095010Z-stage4-zero-3f82e1` completed that diagnostic with the same
built candidate and rendered inputs. With zero generated/verified proposals,
the repeated passage matched and the ordinary continuation was byte-identical
to the saved plain reference, including its natural stop at 60 tokens. Both
responses reported zero proposals. Raw residual export and head priming alone
therefore reproduce this reference; next inspect the first differing verification
decision and prior accepted-prefix resolutions. This does not yet distinguish
verification numerics from a rejected-suffix state defect.

Build `20261019T095412Z-stage4-build-b0cc40` adds an optional host round trace
(`MAGNITUDE_TRACE_GENERATION_ROUNDS`) without changing proposal construction or
acceptance. Its candidate digest is
`ac2db9bba6a9240f1414b48c8e6faa06035aa53d8da42abaf81c4b94044497ea`.
`20261019T095455Z-stage4-trace-2e7d29` reproduced the same differing continuation
and the same 36/26/10 proposed/accepted/rejected counts. It is a diagnostic
completion, not a passed parity gate.

The first differing verification starts at target position 411 with 16 output
tokens already emitted. Its inputs are token 8026 (the anchor) and token 11
(the proposed comma); target selections are `[11, 3804]`. The comma is correctly
accepted, then the target's bonus token 3804 produces ` went`, where plain
produced ` and`. Seven rejected proposals preceded this round. The differing
token is therefore a target selection, rather than an unchecked proposal
published by acceptance. This does not establish whether prior state resolution
or two-row target numerics caused the changed selection. `analysis.json` retains
the exact trace line and the compared outputs.

### Stage 4 target-only batching diagnostic

The evaluation harness now accepts explicit decoder ownership for model loads,
a bound on leading scored positions, and optional exact feature exports. A
width comparison keeps the model context fixed. Placed timing/attribution paths
that currently observe only one owner's trace fail before loading instead of
reporting partial device time. The six harness/design changes are qualified
independently of promoting the MTP implementation.

The canonical `prose-continue/moby-dick/0` probe retained its original scored
assistant answer. Its first 396 rendered tokens and template match the failed
serving request exactly; the full fixture contains 580 tokens. Both widths use
context 581, the same two ranges, prompt group 8 and accepted reuse policies,
with no drafter, lookahead or rejected suffix. Each two-row teacher-forced input
is committed whole. These are bounded diagnostics, not reference-quality or
speed claims; no thermal timing comparison was attempted.

| Diagnostic | Scored positions | Mean / maximum KL, width 1 to 2 | Argmax agreement |
| --- | --- | --- | --- |
| Default affine KV | 395–418 (24 rows) | 0.005291 / 0.042224 nats | 23/24 |
| Dense KV | 395–398 (4 rows) | 0.009330 / 0.022439 nats | 3/4 |

Both changed the argmax at position 397 from token 3896 to 34565. The single-row
winning margin was 0.128477 logits with affine KV and 0.192133 with dense KV.
No measured row exceeded 0.5 nats. The differing row counts prevent comparing
the two means as a codec-quality verdict. Target batching therefore changes
outputs without drafting or rejected-suffix repair, and compressed KV storage
is not required. This does not yet establish the cause of the serving difference
at position 412 or exclude an additional repair problem.

The corrected affine feature probe retained four exact BF16 rows, width 2,560.
All four differ across widths, with RMS differences of 0.077298–0.163667 and
maximum absolute differences of 1.0–1.75. Its logits are bit-identical to the
same four rows of the earlier logits-only probe on both widths. The difference
thus extends to the exported feature path. Packed vocabulary projection also
prepares its own features from final hidden rows, so this export alone does not
identify its internal operands or the first differing decoder operation.

The first feature observer incorrectly read the first row of a request-wide
aggregate for every result, duplicating odd rows of two-row forwards. Its
`observer-issue.json` supersedes those feature measurements. The corrected
observer checks aggregate row count and slices each row at its actual offset;
the original logits were unaffected.

Local evidence under `inference/evaluation/results/`:

- `20261019T100507Z-stage4-build-2a65d7`: target-only probe build, base
  `47f852f5`, dirty digest
  `08bcf307134fb2f0c9bdafd1845390a7eb38afd7737939d48862c280bf24d0fc`.
- `20261019T100640Z-stage4-logits-f180a4`: 24-row affine diagnostic.
- `20261019T101057Z-stage4-dense-2b839c`: four-row dense diagnostic.
- `20261019T101612Z-stage4-features-27eccd`: superseded feature observer.
- `20261019T101823Z-stage4-build-bb2707`: corrected feature build, base
  `47f852f5`, dirty digest
  `059ae8ba8afc817b4511db837b8b884509eea0cda84742fbf3513ded98e5784e`.
- `20261019T101854Z-stage4-features-49cb0b`: corrected feature diagnostic.

Build results retain complete candidate patches. Successful probes retain exact
commands, rendered inputs, logits, analyses, driver and launcher; launchers pin
their own build proof rather than choosing the latest build. These ignored local
artifacts are not bundled with a clone. The initial launcher syntax failure
`20261019T100618Z-stage4-logits-bcc69b` happened before any model work.

The main branch, without the candidate's MTP core, compiled and passed provenance
in `20261019T103223Z-placed-harness-build-ae00bb`: base `6d395edc`, dirty digest
`1f2b4eb3849930817b54e7cd72e126683b1a77aa04bc4473b3f99e99946f0a92`.
`20261019T103247Z-stage4-main-features-3ebbf3` then reproduced all four logits
and exported features byte-for-byte against the corrected candidate probe at
both widths (`main-vs-candidate.json`). This batching difference therefore also
exists without the candidate's MTP implementation. It remains necessary to
localize it and qualify the actual failing serving continuation after correction.

Two earlier main builds compiled successfully but failed provenance because
registry dependencies' generated files under the repository target directory
were classified as synchronized sources without workspace compilation records.
Cleaning release output did not resolve that classification. Moving release
output into the external evaluation cache, retaining its original path through
a symlink, allowed the same unmodified audit to verify it. Both failed records
and controller logs are retained. No provenance or numerical gate was relaxed.

### Stage 4 raw block localization

An isolated diagnostic now captures completed raw block rows at a bounded logical
position range. It flushes and observes producer completions while retaining the
ordinary graph, state and stored-table leases. Reading allocates no device
storage. An optional mixer-boundary export belongs to both checked and prepared
graph layouts, admission charges and graph-sharing identity. These diagnostic
changes remain in the candidate, with full patches in their build results.

Build `20261019T104107Z-stage4-build-dc5106` (base `47f852f5`, dirty digest
`fba25c0a3c72e3c8a35423555ec942aeb2123919697986bbf9a54b73fdc3b6f6`)
and probe `20261019T104221Z-stage4-blocks-818d25` captured position 395 at
every block's input and output. The same four scored positions, context 581,
fixture and two owners were preserved. Both widths' logits match the earlier
probe exactly. All raw rows were finite.

- Block 0's input and the outputs of blocks 0–2 match byte-for-byte across widths.
- Block 3 has identical input but differing output: RMS difference
  **7.140004e-6**, maximum absolute difference **6.954372e-5**.
- Within each width, block 23's output equals block 24's input exactly. The
  initial difference precedes this GPU handoff; `handoff-check.json` retains
  the boundary comparison.
- By block 47's output, the RMS difference is **0.06008935**, with maximum
  absolute difference **0.7457142**. This is a diagnostic row, not a model-quality
  or timing result.

Build `20261019T104639Z-stage4-build-5377c3` (same base, dirty digest
`67faba0d007a6dfd528ae5e452de9e77237fb5fe3cfe13a24bed8e292b3c38ef`)
adds the admitted mixer-boundary export for block 3. Probe
`20261019T104747Z-stage4-middle-66c44f` confirms that this intermediate already
differs: RMS **6.751721e-7**, maximum absolute difference **3.686594e-5**.
Its execution trace identifies blocks 0–2 as recurrent and block 3 as attention.
The first observed difference is therefore inside block 3's attention mixer,
before its feed-forward, rather than first appearing in vocabulary projection.
That mixer includes stream mixing, attention projections, history/attention and
the residual join; the specific operation is not yet identified.

The added export reproduced all 96 earlier raw rows, logits and exported features
byte-for-byte at both widths (`observer-check.json`). The recorded attention
configuration is the same for both row regimes and both widths: MATRIX=1,
PARTS=12, WARPS=4, COLUMNS=1, QREG=1, SLICES=1, STAGES=2. Different selected
parameters are therefore not supported as the explanation for this probe.
This does not establish arithmetic parity inside that configuration. Both runs
are untimed diagnostics with pinned launchers and archived input/output data;
the MTP serving parity gate remains failed.

### Stage 4 attention arithmetic and canonical history joins

The admitted diagnostics now retain mixer input, Q/K/V, indexer projections and
attention output using their producers' descriptors. Build
`20261019T105520Z-stage4-build-ce250f` (base `47f852f5`, dirty digest
`cfe698619ae771e06146062d70d4bab55588a8ff28c81bd9d67616ab869d3ede`)
and probe `20261019T105647Z-stage4-attention-0e524d` localize the first change
at position 395 more precisely:

| Block 3 observation | Across widths 1 and 2 |
| --- | --- |
| Mixer input, projected Q including gate, K and V | Byte-identical |
| Projected indexer query and key | Byte-identical |
| Attention result, 6,144 BF16 values | 16 differ, by 1–2 representable steps |

Attention-result RMS difference is **2.048715e-6**, maximum absolute difference
**0.0001220703125**. The added exports preserve every earlier raw observation,
logits and features exactly (`observer-check.json`). All 30 recorded tuning
unit keys and their chosen configurations also match across widths
(`20261019T104747Z-stage4-middle-66c44f/configuration-check.json`). This narrows
the change to history/preparation/attention rather than Q/K/V projection, but
does not establish an independent model-quality verdict.

**Query-tile experiment, insufficient alone.** Build
`20261019T110211Z-stage4-build-18af2b`, dirty digest
`e5814cf1528add8dae38a587548c63bc638149e6646f22fffe3d8b89c104a1e2`,
assigned one query token per CUDA matrix tile, keeping tokens parallel and
prefill unchanged. Probe `20261019T110346Z-stage4-query-tile-f8ecc0` made all
104 raw observations at position 395 equal across widths; single-row logits,
features and raw observations were unchanged. The four-position mean KL fell
from 0.003611 to 0.000522 nats and argmax agreement became 4/4. Later features
still differed; these are batching diagnostics, not speed or quality gains.

`20261019T110723Z-stage4-trace-9ce653` still failed ordinary serving parity:
35 proposals, 27 accepted, 8 rejected, and 64 output tokens instead of the
plain reference's natural stop at 60. Its repeat fixture matched with 31/31
proposals accepted. Probe `20261019T111106Z-stage4-second-row-b98e90` then
captured position 396. Its blocks 0–2, block 3 input, Q/K/V and indexer
projections match, but the attention result differs (RMS **9.091388e-6**,
maximum **0.00048828125**). Isolating query tiles alone is therefore insufficient.
The original grouped CUDA kernels were restored before the next experiment.

**Canonical history correction, candidate only.** The target builder appended
tentative cached rows to accepted ranges without coalescing their common
boundary. This could give verification `[0,395), [395,396)` where straight
advancement supplied `[0,396)`. The CUDA attention partitioning works over
spans, so logically identical histories can acquire different physical work.

The candidate now uses one append/coalesce helper for causal, indexer and Shared
reads. It preserves logical order, joins only adjacent ranges within one slab,
and leaves the read rows unchanged. Stored-history launch validation uses the
same canonical preparation. Shared validation preserves the exact accepted
prefix and permits its last span to grow only into rows reserved by the advance
in that same slab. No precision tolerance or acceptance rule changed.

The initial producer-only probe `20261019T111704Z-stage4-coalesce-2a72b7`
preserved single-row output but correctly stopped at launch validation, whose
expected prepared history still used the old split form. It is a failed probe,
not a parity result. Build `20261019T112209Z-stage4-build-cea316` includes the
coordinated producer/validator correction, with dirty digest
`9d00a1cc0c2ac4e3216b5404aac8a39384754a22751e297249204ece9f41212d`.

`20261019T112330Z-stage4-coalesce-89b10f` passes the bounded comparison:
all **208 raw observations** at positions 395 and 396 match across widths;
all four logits and exported feature rows are byte-identical; **KL is zero**.
Single-row logits and features remain byte-identical to the saved control
(`byte-parity.json`). This demonstrates the correction on this canonical prefix,
with the original grouped kernels; it is not whole-continuation qualification.

The actual MTP diagnostic `20261019T112543Z-stage4-trace-9ee35f` remains failed
for ordinary parity: 35/27/8 proposed/accepted/rejected, 64 output tokens, while
the repeated passage matches at 31/31. The first differing round is still at
position 411 after 16 emitted tokens: input `[8026,11]`, target selections
`[11,3804]`. The comma is accepted, then bonus token ` went` differs from plain
` and`; **five rejections** precede this round. The canonical book answer used
for teacher forcing is not the saved generated continuation. Exact parity on its
first four scored positions therefore does not isolate the remaining serving
failure from later batching differences or rejection-dependent state handling.

All cited artifacts are under the local results directory. Successful runs have
pinned launchers and exact input/output evidence; builds retain complete patches.
The history correction and MTP implementation remain unpromoted, and timing
remains held until the actual serving gate passes.

### Stage 4 failed text replay and combined serving gate

The saved plain output was rendered as an explicitly identified diagnostic chat
fixture in `20261019T113504Z-stage4-greedy-prefix-7ac07e`. Build
`20261019T113339Z-stage4-build-8b1643` adds an explicit logits-harness model
context reservation, allowing the serving capacity of 1,024 without changing
the input. Its dirty digest is
`559cb4a268429619934feaa2675d4cf6910ef10d55e1ee90e7a33db6c8d83abd`.

The rendered prompt's first 396 tokens match serving, and positions 411–413
contain `[8026,11,321]`. However, **the single-row replay fails its own saved
greedy control at position 397**: the rendered answer expects token 34565 and
the target selects 3896. The run terminated with code 1 and records
`control-issue.json`. Re-rendered text does not establish the actual sampled
token path. The source of this mismatch needs sampled-token evidence; this
probe cannot isolate rejection-dependent repair. Its comparison at position
412 must not be described as proof of a repair defect or an exact serving-prefix
execution comparison.

The later generated-prefix observation below confirms that all sixteen rendered
input IDs at positions 396–411 match native generation. Retokenization is
therefore not the explanation for this bounded prefix. The old logits walk
prefilled 395 rows and then decoded its first scored row; native generation
prefills all 396 prompt rows and selects at the prompt boundary. Its output
demand also differs. Matching token IDs alone does not establish matching
numerical state or execution boundaries.

Both candidate corrections were then combined: canonical history joins and
one query per CUDA matrix tile, with queries still parallel. Build
`20261019T114027Z-stage4-build-2b0009`, dirty digest
`24f3085f7159b3706b43425a1daec01e044e3d8a6fc9742644a4ddc3113bde94`,
completed. The full original serving gate
`20261019T114235Z-stage4-combined-heldout-a58779` **failed**:

| Fixture | Frozen plain control | Combined MTP candidate | Exact parity |
| --- | --- | --- | --- |
| Repeated passage | 64 output tokens, length limit | 64; 31/31 proposals accepted | Pass |
| Ordinary continuation | 60 output tokens, natural stop | 64, length limit; 35 proposed, 27 accepted, 8 rejected | Fail |
| Tool call | 53 output tokens, tool stop | 53; 26/26 proposals accepted | Pass |

Both tool outputs passed BFCL validation and called `calculate_triangle_area`
with exactly `{"base":10,"height":5,"unit":"units"}`. Generated call IDs
are excluded from parity. The candidate's ordinary text again selects ` went`
where plain selects ` and`. Its round trace reproduces position 411 inputs
`[8026,11]`, target selections `[11,3804]`, after five prior rejection rounds
(`round-analysis.json`). Combining the corrections does not qualify MTP.

The run retains all controls, stop reasons, outputs and checks, plus a launcher
pinned to its candidate build proof. It is instrumented qualification with no
thermal timing gate and supports no speed claim. Both corrections and the MTP
implementation remain candidate-only.

### Stage 4 generated-prefix control isolates attention configuration

The harness now has an opt-in `target-compare` command. It renders one chat
fixture, generates a prefix through actual greedy selections, then compares two
sequential rows with one paired target forward on a freshly regenerated prefix.
It requires the complete sampled prefixes to match before comparing selections.
Both walks use the same context capacity, selection-only output demand and
admitted decode capacity. The paired continuation comes from the first walk's
sampled token; no persisted token list is supplied as model input. This is a
bounded batching diagnostic, not independent model-quality evaluation.

With both experimental corrections still applied, build
`20261019T115715Z-stage4-build-552f7b` (dirty digest
`4d2c02ffad4d6a4d5855f911a69772ae73d4c9cb7be646cb941b4ad6ac587eb7`)
and run `20261019T115825Z-stage4-generated-prefix-291e13` reproduced the
sixteen serving-prefix tokens, and sequential/paired selections matched.
However, **both selected `went`**, so this did not preserve the frozen plain
serving control. Run `20261019T120113Z-stage4-current-plain-control-a5d910`
then confirmed that the same candidate server diverged with drafting completely
disabled: the repeated passage matched, but ordinary continuation reached its
64-token limit instead of the frozen control's natural stop at 60.

The query-tile experiment changed the kernel implementation identity and its
stored tuning choices. The candidate's single-row attention used the canonical
fallback where the frozen control used a qualified configuration. A successful
bounded batching comparison under that changed plain path does not establish a
serving fix. **The query-tile patch has been removed**, including its design
requirement. Canonical history joins remain in the candidate.

Build `20261019T120404Z-stage4-build-2c75d0`, dirty digest
`4fe9e83d36eaebdac666b79500ce409c04db323ef1beb49712b26ca6fb142165`,
restores the original CUDA kernels. Run
`20261019T120525Z-stage4-original-generated-prefix-8902cd` now establishes
the relevant failure without MTP or rejected suffixes:

| At the generated serving prefix | Sequential target | Paired target |
| --- | --- | --- |
| Position 411, input 8026 | Selects comma, token 11 | Selects comma, token 11 |
| Position 412, input 11 | Selects ` and`, token 321 | Selects ` went`, token 3804 |

Both fresh walks generated the same sixteen tokens at positions 396–411,
matching the actual earlier serving trace. Context capacity is 1,024 and the
native serving prompt is exactly 396 tokens. The run correctly terminated
with a failed parity result. This shows that the differing selection does not
require rejected-prefix repair; it does not rule out other lifecycle defects.

The earlier observation that both loads had the same 30 tuning records was
insufficient: it compared available units, while the two row widths invoke
**different units within that same set**. Their actual affine decode choices are:

| Parameter | Single-row unit `M=1..2` | Multi-row unit `M=2..9` |
| --- | ---: | ---: |
| MATRIX | 1 | 1 |
| PARTS | 48 | 12 |
| WARPS | 1 | 4 |
| COLUMNS | 4 | 1 |
| QREG / SLICES / STAGES | 1 / 1 / 2 | 1 / 1 / 2 |

To isolate this difference, build
`20261019T120658Z-stage4-pinned-build-cee893` enables the existing harness-only
`pinned-tuning` feature. Run `20261019T120820Z-stage4-pinned-prefix-7a48cb`
replays all 30 observed units and changes **only the multi-row affine attention
unit** to the single-row parameters. All 59 replay events name recorded choices;
no unit falls back to an unrecorded default. The generated prefix and sequential
selections remain unchanged. **Both paired selections now match**, including
token 321 at position 412. `pin-change.json`, `replay-events.json` and
`qualification.json` retain the complete control and change.

This isolates an attention-configuration cause at the failing generated round.
The development replay does not run normal tuning qualification and cannot
serve as a production fix or a speed claim. The production change must preserve
the target's arithmetic across these row widths, with ordinary resource and
numerical admission, then pass actual serving and lifecycle gates before timing.

The reusable harness command, explicit logits context capacity, and the missing
optional precision-probe constructor fields are now retained on the working
branch. Release build
`20261019T121135Z-generated-prefix-harness-build-8b4e53` compiled the harness
with `precision-probe` enabled and passed the unchanged provenance audit (base
`044de819`, dirty digest
`50d8446b9f101168d4913bb3b24cbf30faa8246e7bbdb2d7c4ceab780f77828e`).
The MTP implementation and history correction remain candidate-only. All run
artifacts above are local under `inference/evaluation/results/`; model jobs were
untimed diagnostics under the machine lock. No speed improvement is claimed.

### Stage 4 shared decode unit passes strict serving qualification

The candidate now prepares one CUDA affine decode tuning unit for query groups
of 9–16 heads per K/V head and widths at most 256. Both single-row decoding and
verification bind that choice. The unit's normal numerical contract includes
all served row and history shapes; other backends keep their row bands. Its
initial search hint comes from head geometry, and all declared configurations,
native-default numerical admission and the fixed tuning budget remain intact.
The original CUDA kernel sources are unchanged. This uses the normal tuner,
without a development pin or an independent startup qualification allowance.

Build `20261019T122031Z-stage4-build-87d1df` completed (base `47f852f5`,
dirty digest `da537eff49fa7ed66bd13fb41191db2a728fbc3c143534ecb9816771c02d37b5`).
The new `M=1..9` unit measured 84 configurations and passed its numerical
coverage, including untimed witnesses. Its 30 points cover row classes 1, 2, 4
and 8, single and multiple sequences, and histories 0, 256 and 1,024. The
target-only diagnostic's performance demand was single-row; zero-demand
verification points still supplied required numerical coverage. The chosen
parameters are MATRIX=1, PARTS=48, WARPS=2, COLUMNS=4, QREG=0, SLICES=1,
STAGES=2. This is a newly qualified shared choice, not reuse of the old
single-row qualification as evidence for wider rows.

`20261019T122157Z-stage4-shared-prefix-f881f0` passes without a pin. Its sampled
prefix and sequential selections remain identical to the saved original-kernel
control. Both sequential and paired execution select comma at 411 and token
321 (` and`) at 412. The run also requires the shared unit's qualified status
and passing coverage; a matching point under an unqualified fallback cannot pass.

The full serving comparison
`20261019T122452Z-stage4-shared-strict-heldout-c073bf` passes exact text/tool
parity **and** prompt/output/total token counts and stop reasons:

| Fixture | Output tokens and stop | MTP proposed / accepted / rejected |
| --- | --- | --- |
| Repeated passage | 64, length limit | 31 / 31 / 0 |
| Ordinary continuation | 60, natural stop | 34 / 24 / 10 |
| Tool call | 53, tool call | 26 / 26 / 0 |

Both tool outputs passed BFCL validation with exactly
`calculate_triangle_area({"base":10,"height":5,"unit":"units"})`.
Ordinary continuation now matches the frozen Stage 2b plain control despite ten
rejected proposals. This qualifies these requests, not arbitrary prompts or
independent distributional quality. The run is instrumented and untimed.

The declared comparison uses untraced A/B/B/A rounds in one thermal-gated lease,
with repeated passage and ordinary continuation, two warmups per fixture per
launch and two measured passes per side and fixture. Ordinary generation time
per output token is primary; require at least 3% and a difference beyond both
observed spreads. Prefill, complete model time, acceptance and memory costs
remain separate results. Results and the resulting disposition follow.

### Stage 4 lifecycle and controlled performance decision

`20261019T123227Z-stage4-shared-lifecycle-8d6b29` passes on the same candidate:
cold, retention, reuse and recovery outputs are identical 64-token expected
prefixes, with 31/31 proposals accepted each. Usage and stop reasons are checked.
Reuse and recovery each reuse 757 prompt tokens. The client closes cancellation
after four content events, before terminal delivery; recovery remains exact and
the health check passes. This covers retained-prefix recovery on the full-accept
fixture. Rejected rows are exercised by ordinary continuation in the separate
strict serving gate; these runs do not establish retained-prefix behavior for
every rejection history.

`20261019T123708Z-stage4-comparison-626259` completes the declared thermal-gated,
untraced A/B/B/A comparison. **All 24 responses match the qualification controls**
in text, usage and stop reason. Both conditions use decode-reuse seeding and
replacement, no retained prefixes, managed host cache off and context 1,024.
The OS file cache is not dropped. Loading and warmups are excluded from the
following model times, which include the candidate's head, verification and
repair work. These compare the complete Stage 4 variant with the frozen Stage
2b plain build, not just a method flag on one binary.

| Fixture / condition | Mean prefill | Mean generation | Generation ms/output token | Mean total model time |
| --- | ---: | ---: | ---: | ---: |
| Repeat, plain | 3.700 s | 3.367 s | 52.610 | 7.067 s |
| Repeat, MTP | 4.419 s | 2.629 s | 41.083 | 7.048 s |
| Ordinary, plain | 3.084 s | 2.260 s | 37.663 | 5.344 s |
| Ordinary, MTP | 3.746 s | 2.193 s | 36.555 | 5.939 s |

Repeated-passage generation improves **21.911%**, exceeding both observed
generation spreads (129.721 ms plain, 63.277 ms MTP). Its 718.769 ms prefill
increase absorbs almost all the 737.741 ms generation saving: total model time
changes by only **−0.268%**, within the observed total-time spreads.

Ordinary generation improves **2.940%**, a 66.438 ms difference exceeding the
observed spreads (34.057 ms plain, 2.134 ms MTP), but **below the declared 3% bar**.
It must not be rounded up to a pass. Prefill increases 661.188 ms (**21.437%**),
and total model time regresses **11.129%**, or 594.750 ms. Its accepted/proposed
ratio is 24/34, versus 31/31 on the repeated passage. Two measured passes per
side and fixture support this scoped decision, not a broad workload claim.

Measured ordinary-generation delivery windows show **zero process-attributed
physical read bytes on both sides**. Logical reads average 3.878 GB plain and
4.685 GB MTP: roughly **20.8% more logical traffic** for the candidate. Those
windows include lookahead and are not exact device-step boundaries. They give
no evidence that additional host RAM would remove this warm-generation cost.
They do not separately attribute RAM copies, uploads, conversion or head compute.

The sampled allocation census also warrants later placement investigation.
Plain uses 200/203 cache records per layer on owners 0/1; MTP uses 135/175.
MTP carries 2,668,104,704 auxiliary bytes on GPU 0. Its minimum sampled device
headroom is about 3.62/3.83 GiB, versus 2.63/2.14 GiB plain. Headroom is not an
authorization to allocate all remaining bytes: startup/import peaks, graph and
state reserves still apply. The range cache budget is inherited from full-model
assessment, so Stage 7 should examine exact per-owner peaks before changing it.
No safe additional cache capacity or speedup has been established here.

Source inspection rules out one proposed shortcut: causal-only MTP priming
already stops after K/V publication and binds no post-attention/FFN/readout
weights (`native_head`'s zero-step graph). The added prefill time cannot simply
be blamed on an unnecessary full-head FFN pass. Further attribution is required.

**Decision:** preserve the qualified implementation as an experimental capability
on [the MTP experiment branch](https://github.com/thebadking/magnitude/tree/two-gpu-mtp-experimental).
Do not promote this fixed-budget configuration as a general serving optimization;
keep the accepted plain profile as the control for Stage 5. Repeated generation
has a real scoped benefit, while ordinary short-request total time regresses.
Longer-output amortization is a possible future experiment, not a measured gain.
Maximum context, smaller-model single-GPU regression and cumulative comparison
remain Stage 8 obligations before broader configuration claims.

The measured source identity remains `47f852f5` plus the full `da537eff…` digest
above; publication does not relabel the experiment as a measurement of a later
commit. Complete patches, input/output controls, thermal gate, per-pass timing,
I/O windows and memory census are retained under the comparison directory.
An earlier launch, `20261019T123507Z-stage4-comparison-cc4bb0`, stopped during
setup because the driver lacked an import; it performed no model measurements.
That failed setup and its exact script are retained separately.

## Immediate next action

Continue **Stage 5** with the frozen Stage 2b plain control. Use the current
warm I/O evidence to distinguish source service, copies/transfers, conversion
and resident execution before choosing one SM86 kernel or representation change.
Inspect actual packed formats and shapes, preserve the fixed tuning budget and
normal numerical gates, and benchmark the justified change under matched work.

Keep Stage 4's source and artifacts as the experimental MTP control. Its current
scope is the catalog's causal head with its own history; target-reading heads,
vision and separate block drafters need other placement contracts. Queries remain
parallel; the query-tile experiment is discarded. Do not serialize verification
or relax precision/greedy parity to obtain a pass. Study the observed per-owner
cache/headroom gap during Stage 7 before allocating more cache, then complete
Stage 8's affected context and single-GPU qualification and cumulative comparison.
Stage 3's repeated-miss candidate is rejected; a differently conditioned predictor
remains deferred. Full CUDA graph attribution is still unavailable.
