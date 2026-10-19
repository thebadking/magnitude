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
| 2 | Trace-informed GPU expert retention | Ready: protect useful decode residency during prompt seeding | 1 decision |
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

## Immediate next action

Begin **Stage 2a: prompt seeding informed by decode reuse**, from the accepted
synchronous-transfer baseline. Preserve the rejected transfer patches and their
measurements. Complete and qualify the new policy, then benchmark it before
trying a separate replacement-policy change. Full CUDA graph attribution remains
unavailable; do not imply a complete wait/arithmetic breakdown.
