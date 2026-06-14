# Bugfix Implementation Plan — Vector Similarity & Retrieval

**Scope:** This plan covers two defects only:

1. The cosine similarity computation in `search.ts` is mathematically wrong.
2. Vector retrieval is gated by lexical (FTS5) recall and runs as a brute-force JS
   scan over a small candidate set, so it cannot scale and cannot surface
   semantically-relevant-but-lexically-disjoint content.

**Explicitly out of scope** (do not implement here — a separate plan will follow):
session-log chunk ingestion, the thinking-phase indexing decision, and the
usefulness/feedback loop. The only ingestion-adjacent work permitted here is a
one-time **migration/backfill of vectors for nodes that already exist**, because the
new retrieval path needs a populated index to be testable.

Target files: `search.ts`, `db.ts`, `index.ts`, `package.json`, and the `test/`
suite (`search.test.ts`, `db.test.ts`).

---

## Part A — Fix the cosine similarity math

### A.1 The defect

In `search.ts`, `cosineSimilarity(a, b)` accumulates the second magnitude term from
the wrong vector:

```ts
for (let i = 0; i < a.length; i++) {
  dotProduct += a[i] * b[i];
  normA += a[i] * a[i];
  normB += a[i] * a[i];   // BUG: uses a[i], should use b[i]
}
const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
```

Because `normB` is computed from `a`, `magnitude` collapses to
`sqrt(normA) * sqrt(normA) = ‖a‖²`, so the function returns `dot(a,b) / ‖a‖²`
instead of `dot(a,b) / (‖a‖·‖b‖)`. This only coincides with true cosine when
`‖a‖ == ‖b‖` (e.g. both vectors unit-normalized). The embeddings returned by the
LMStudio endpoint for `nomic-embed-text-v1.5` are **not guaranteed to be
unit-normalized**, so the current ranking is using a distorted score.

This has been masked for curated nodes because BM25 (weight `0.4`) dominates the
composite and curated facts share lexical tokens with queries. It will **not** stay
masked once retrieval leans on semantic similarity, which is the whole point of the
upcoming session-chunk work — hence fixing it now.

### A.2 The fix

Correct the loop so each norm uses its own vector:

```ts
for (let i = 0; i < a.length; i++) {
  dotProduct += a[i] * b[i];
  normA += a[i] * a[i];
  normB += b[i] * b[i];
}
```

Keep the existing guards: the dimension-mismatch early return (`return 0`), the
zero-magnitude guard (`if (magnitude === 0) return 0`), and the final clamp to
`[0, 1]`. The clamp treats anti-correlated vectors as "no similarity" — that is a
deliberate product choice for this ranker; leave it as is.

### A.3 Normalization (optional, not required)

You do **not** need to normalize stored embeddings to make A.2 correct. If you later
want the JS path to skip the norm computation entirely (similarity becomes a plain
dot product), you may normalize on store, but that is a micro-optimization, not part
of this fix. Default: leave embeddings raw and rely on the corrected formula.

### A.4 Tests (add to `test/search.test.ts`)

- Identical vectors → similarity `1.0` (within float tolerance).
- Orthogonal vectors → `0.0`.
- Anti-parallel vectors → `0.0` (clamped).
- **Non-unit-length vectors** that would expose the old bug: e.g.
  `a = [3, 0]`, `b = [0, 4]` → `0`; `a = [1, 2]`, `b = [2, 4]` (parallel,
  different magnitudes) → `1.0`. The second case fails under the old code and
  passes under the fix — make it explicit so the regression can never silently
  return.
- Dimension mismatch → `0` (no throw).

---

## Part B — Fix candidate gating and the brute-force scan

### B.1 The defect

In `search.ts`, `searchHybrid()` builds its candidate set from FTS5 only:

```ts
const ftsResults = db.searchFTS5(query, maxResults * 3, ...);   // lexical candidates
// ... vector similarity is then computed *within* ftsResults
```

and on an FTS miss falls back to `db.getFallbackCandidates(50, ...)`, which is
ordered by `(frequency + access_count) DESC, last_accessed DESC`. Consequences:

- **Vector search can only re-rank what BM25 already found.** Anything lexically
  disjoint from the query is unreachable, regardless of semantic relevance.
- **The fallback is degenerate for fresh content.** For nodes with zero
  frequency/access (which is every freshly ingested chunk), the fallback reduces to
  "50 most recent nodes," which is meaningless for semantic recall.
- **It is a brute-force JS scan.** Acceptable at the current curated-node scale
  (hundreds), but it is exactly the path the session-chunk tier (tens of thousands)
  would inherit, where it breaks on both correctness (gating) and latency.

There is also a query-time inefficiency to remove: `searchHybrid` calls `batchEmbed`
on un-embedded **candidate nodes** mid-query, then re-reads them via `getVector` in a
per-candidate loop. Node embeddings belong at write/index time, not query time.

### B.2 Target architecture

Make vector search a **first-class candidate source**, unioned with FTS, rather than
a re-ranker gated by FTS. Use an ANN/KNN index so this scales to the chunk tier that
follows.

```
query ──┬─► FTS5 (BM25)        ─► candidate IDs (lexical)
        └─► vec0 KNN (cosine)  ─► candidate IDs (semantic)
                       │
                 union of IDs
                       │
        hydrate nodes + compute existing composite score
        (bm25 ⊕ vector ⊕ frequency ⊕ freshness) ─► sort ─► top-N
```

Use **`@photostructure/sqlite-vec`** (the maintained Node fork already named in the
README and dependency list) for the KNN index. It provides `vec0` virtual tables
with in-C KNN, ships prebuilt binaries, and at 768-dim handles ~100k vectors with
sub-100ms KNN — comfortably ahead of where the chunk tier will start.

> **Version check before coding:** `vec0` is pre-1.0 and its surface has shifted
> across releases. Confirm against the installed package's README: (a) the exact
> loader call, (b) whether a `TEXT PRIMARY KEY` column is supported (fallback:
> key the vec table by `nodes.rowid` and JOIN), and (c) that `distance_metric=cosine`
> is accepted in the `vec0` table definition. Treat the snippets below as the shape,
> not verbatim API.

### B.3 Step-by-step

**1. Dependency.** Add `@photostructure/sqlite-vec` to `package.json` `dependencies`
and install. (`better-sqlite3` already supports loadable extensions.)

**2. Load the extension and create the index** — in the `KnowledgeGraphDB`
constructor in `db.ts`, after the DB is opened and pragmas are set, before/with the
existing `SCHEMA_SQL` execution:

```ts
import * as sqliteVec from '@photostructure/sqlite-vec';

// after `this.db = new Database(...)` and pragmas:
let vecAvailable = false;
try {
  sqliteVec.load(this.db);          // confirm exact loader name vs installed pkg
  vecAvailable = true;
} catch (err) {
  console.warn('[kg-memory] sqlite-vec unavailable, vector KNN disabled:', (err as Error).message);
}
this.vecAvailable = vecAvailable;   // store on the instance

if (vecAvailable) {
  this.db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(
      node_id   TEXT PRIMARY KEY,
      embedding float[768] distance_metric=cosine
    );
  `);
}
```

Keep the dimension (`768`) sourced from config rather than hard-coded if practical;
at minimum leave a comment tying it to `embeddingModel`.

**3. Keep `node_vectors` as canonical storage; treat `vec_nodes` as a derived
index.** Do **not** drop the existing `node_vectors` table. It holds the raw bytes
plus `model`/`dim` provenance, which you need to rebuild the index after an embedding
model change. `vec_nodes` is the fast queryable projection of it.

Update `db.storeVector(nodeId, embedding, model)` to write **both**: the existing
`node_vectors` upsert, and an upsert into `vec_nodes` (guarded by `this.vecAvailable`
and a `model === config.embeddingModel && embedding.length === 768` check). Bind the
vector as the compact float32 binary the code already produces:

```ts
const buf = Buffer.from(new Float32Array(embedding).buffer);
// node_vectors upsert (existing) ...
if (this.vecAvailable) {
  this.db.prepare(
    'INSERT OR REPLACE INTO vec_nodes(node_id, embedding) VALUES (?, ?)'
  ).run(nodeId, buf);
}
```

Mirror this in `deleteNode`: delete from `vec_nodes` alongside `node_vectors`.

**4. Add a KNN method to `db.ts`:**

```ts
/** Returns nearest node_ids by cosine distance. Empty array if vec unavailable. */
knnSearch(queryEmbedding: number[], k: number): { nodeId: string; distance: number }[] {
  if (!this.vecAvailable) return [];
  const buf = Buffer.from(new Float32Array(queryEmbedding).buffer);
  return this.db.prepare(`
    SELECT node_id AS nodeId, distance
    FROM vec_nodes
    WHERE embedding MATCH ? AND k = ?
    ORDER BY distance
  `).all(buf, k) as { nodeId: string; distance: number }[];
}
```

Use the `AND k = ?` form (works on all SQLite versions); the bare `LIMIT` form
requires SQLite ≥ 3.41. Cosine **similarity = 1 − distance** under
`distance_metric=cosine`.

**5. Rewrite candidate gathering in `searchHybrid()`** so FTS and vector are unioned.
The existing scoring layer (`computeCompositeScore` and the per-term helpers) does
**not** change — only how candidates are collected and how each term is populated:

- Embed the query once (`getQueryEmbedding`) — keep this.
- Gather FTS candidates: `db.searchFTS5(query, maxResults * 3, ...)` as today.
- Gather vector candidates: `db.knnSearch(queryEmbedding, maxResults * 3)`, then
  hydrate those node IDs (respecting the `categories`/`subcategories` filters — apply
  them after KNN, or use vec0 metadata-column pre-filtering if you add the columns).
- **Union by node ID.** For each unique candidate compute:
  - `bm25Score`: from the FTS hit if present, else `0`.
  - `vectorScore`: `1 − distance` from the KNN hit if present; if the node was a
    pure-FTS hit with no KNN entry, look up its stored vector via `getVector` and
    score with the (now-correct) `cosineSimilarity`, else `null`.
  - `frequencyBoost`, `freshnessScore`: unchanged.
- Run `computeCompositeScore(...)` over the union, sort, slice to `maxResults`.
  The `vectorScore === null` redistribution branch already handles candidates with no
  vector — leave it.

**6. Remove query-time embedding of candidate nodes.** Delete the `batchEmbed` call
on un-embedded candidates inside `searchHybrid`. Node vectors are populated at
write/index time (existing nodes via the B.4 backfill below; future session chunks via
the separate ingestion plan). The only embedding that happens in the query path is the
query vector itself.

**7. Graceful degradation (preserve the "fail soft" philosophy).** When
`this.vecAvailable` is false **or** `getQueryEmbedding` returns `null`, fall back to
the FTS-only path that already exists (now benefiting from the corrected math wherever
`cosineSimilarity` is still used). Note in a code comment that this fallback is
acceptable only at the curated-node scale and that `vec_nodes` is required before the
chunk tier lands — do not silently rely on it there.

### B.4 One-time migration / backfill (existing nodes only)

This is the only ingestion-adjacent work in scope, and it covers **already-stored
curated nodes**, not session logs.

Add an idempotent backfill, run once on startup from `index.ts` (guarded so it is a
no-op when already complete — e.g. skip if `vec_nodes` row count == `node_vectors`
row count):

1. For every row in `node_vectors` whose `model`/`dim` match the active config,
   `INSERT OR REPLACE` it into `vec_nodes` (no re-embedding needed — reuse stored
   bytes).
2. For nodes with **no** vector at all, batch-embed their `content` via the existing
   `batchEmbed`, store via the updated `storeVector` (writes both tables). Bound and
   batch this; do not block startup on a large synchronous loop — reuse the batching
   already present in `batchEmbed` and yield between batches.

Because `storeVector` now writes both tables, steady-state writes need no separate
migration; this backfill exists only to populate the index from history.

### B.5 Tests

- `db.test.ts`: round-trip a vector through `storeVector` → `knnSearch` returns it at
  `distance ≈ 0`; `deleteNode` removes it from `vec_nodes`; `knnSearch` returns `[]`
  when `vecAvailable` is false.
- `search.test.ts`: a node that is **semantically related but shares no query tokens**
  is retrieved via the vector path (it would be invisible under the old FTS-gated
  candidate set — this is the core regression test for B.1); FTS-only and
  vector-only candidates both appear in the unioned, scored result; ranking is stable
  when the embedding endpoint is mocked unavailable (degrades to FTS).
- A test asserting `searchHybrid` issues **no** node-embedding calls in the query path
  (only the single query embedding), to lock in B.6/step 6.

---

## Risks & confirmations

- **`vec0` API drift** — confirm loader name, TEXT-PK support, and
  `distance_metric=cosine` against the installed version before coding (see B.2 note).
  If TEXT PK is unsupported, key `vec_nodes` by `nodes.rowid` and JOIN; the codebase
  already depends on `rowid` stability for external-content FTS, so this is consistent.
- **Dimension coupling** — the `float[768]` width is fixed at table-creation. If the
  embedding model ever changes dimension, `vec_nodes` must be dropped and rebuilt from
  `node_vectors`. Leave a comment and, ideally, assert `embedding.length === 768`
  before inserting.
- **Filter semantics** — category/subcategory filters currently live in the SQL of
  `searchFTS5`/`getFallbackCandidates`. Decide whether to replicate them as vec0
  metadata-column pre-filters or apply them post-KNN; post-KNN is simpler and fine at
  current scale, but note that it can under-fill `k` when filters are selective
  (request a larger `k` to compensate).

## Acceptance criteria

1. `cosineSimilarity` returns true cosine for non-unit vectors (A.4 tests pass).
2. A semantically-relevant, lexically-disjoint node is retrievable (B.5 core test
   passes) — i.e. vector search is no longer gated by FTS.
3. `searchHybrid` performs no node embedding in the query path.
4. With sqlite-vec disabled or the endpoint down, search still returns FTS-ranked
   results without throwing.
5. Existing tool/hook tests continue to pass; the composite-score weighting behaviour
   is unchanged for nodes that have vectors.

## Deferred to the next plan

Session-log discovery and chunking, the thinking-phase include/exclude decision, the
chunk feedback/usefulness loop, and any change to how `before_agent_start` injection
selects nodes. The retrieval foundation built here is what those will plug into.
