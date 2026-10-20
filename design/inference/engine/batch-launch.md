---
applies_to:
  - inference/engine/batching/**
  - inference/engine/state/**
  - inference/engine/executor/**
---

# Validated numerical launches

Batching owns device-independent row semantics. It validates slot order, row mappings,
coordinates, visibility, destinations, demands, selection controls, and physical capacity class
once, then produces an opaque domain-specific batch. A row's visible history spans keep the
history's logical order and must not share a row; their addresses need not ascend. The state
store coalesces logically adjacent spans within one slab, and batching preserves its boundaries
so no span crosses a slab. The span class limit is the largest span bound of the loaded store's
history domains, each `ceil(row limit / rows per slab) + 16` (see
[state transactions](state-transactions.md)).
Joining accepted history with tentative rows coalesces physically adjacent
ranges within one slab, in logical order, for causal, indexer and Shared reads.
The join preserves the canonical spans of the equivalent committed history.
Causal target rows using a compressed cache include their earlier reserved destinations in
visible history after ordered cache preparation; their own row remains dense fresh input.
Declared bidirectional media and separate draft blocks retain their own fresh-row semantics.
Its row tables are the only upload source;
programs encode each graph's inputs from them directly, with no separate packed control image.
Selection masks are shared with their producer rather than copied into the batch, and an
unconstrained row carries no mask.

Execution joins a validated batch with owned state advances, conditioning, workspace, and output
leases into one opaque launch. Its constructor checks only cross-domain facts: resource identity,
slot-to-advance cardinality and row alignment, conditioning references, and planned lease class.
Programs accept the launch as one value and retain it in their submission until completion.
Placed launches acquire every owner's graph and transfer leases during admission;
ordinary launches and speculative successors use the same availability checks.
Each prompt-group member owns its complete reservation before the next is built.
Submission consumes those leases and never acquires missing peer capacity.
A layer-major prompt group derives every chunk's preceding-token window from
a private history extended by the known earlier chunks. The first chunk has not
yet been submitted while the group is constructed; its tokens cannot be obtained
from the published fed-token history. Only successful submission and later
claims publish those feeds. A refused or orphaned group cannot advance the
accepted token history.
In a layer-major target group, all entries execute before the decoder blocks.
Kept token rows therefore have one disjoint region per admitted chain member;
an entry cannot overwrite another member's n-gram inputs. Certification and
binding charge the complete group storage, and both single-device and placed
submission select the region by chain index. Later groups reuse the regions in
the device queue's submission order.
Every streamed layer publishes its own launch sequence before routing, including
small prompt tails that use the decode graph. A layer-major group alternates
launches at each layer; no layer may inherit another member's shared step word.
A paired target can continue a device selection when its first owner has
admitted kept-token storage and its entry supports selected tokens. Only the
first chain member may consume a predecessor selection, produced by the final
owner. Before its entry, token and status rows are copied into that member's
first-owner region after producer completion. The launch retains the producer
output lease. Embedding selected-row tracking and stored-table notices remain
part of the ordinary claim checks; moving a selection does not imply that its
embedding row was available. A model without this return storage cannot enable
paired lookahead.
A successor-dependent head's device entry can join one independently owned prior feature with
the leading features of the current target chunk. The join uses planned head scratch and
preserves activation storage exactly; it does not read the features through the host.
When the target producer is on another device, transport first copies the exact
conditioning representation into an admitted independent input slot on the head
owner. Bounded host staging is permitted for that transport; it performs no
numerical conversion. Producer output leases and consumer transfer slots remain
owned until their respective physical work completes. The local join retains
the same successor and previous-row semantics.
A program whose attention graphs have a class that lists history row tiles derives the list from the
batch's visible spans per history domain (`magnitude_batching::history_tiles`: distinct, ascending,
`-1` padded 256-row tiles) and selects the smallest such class the list fits; a launch whose rows see
more tiles than the largest runs the class that lists none. The choice changes neither admission nor
the batch. Listing classes begin at one tile and increase geometrically through the legal
per-request bound. Selecting the smallest fitting class keeps capacity below twice the active
tile count even for short and sliding histories; a fixed minimum class must not force extra
history rounds on a smaller active extent.

Head feature projection is a distinct checked launch because it consumes retained features and
selection controls without advancing sequence state. Its requests, feature extents, domain,
selection masks, and aggregate row capacity are validated together. It uses the head workspace
and output leases and follows the same submit, complete, and reconcile lifecycle as head forward.

Batching does not depend on execution or contain device values. Execution does not reinterpret
independent arrays to reconstruct row alignment inside a native program.

An opt-in block diagnostic may retain exact raw input and output rows over a bounded
logical-position range. It observes all queued producer completions before reading
retained tensors without changing numerical state. Additional intermediate exports
belong to the sealed graph's layout, admission charges and sharing identity;
the observer allocates no device storage during submission.
Named intermediate exports use their producers' actual descriptors rather than
an independent shape or representation definition.
Exports retain the representation, shape, logical position, validated row mapping
and owner. Duplicate output identities fail instead of silently replacing evidence.
Diagnostic synchronization is excluded from production timing claims.

## Acceptance criteria

- Invalid row or slot relationships cannot be represented by a validated batch.
- No visible span crosses a slab boundary, and a batch cannot exceed the loaded model's span bound.
- A launch cannot pair a row with another request's state or resource lease.
- All program inputs remain owned until physical completion and reconciliation.
- No program accepts parallel unchecked row, state, conditioning, or demand arrays.
