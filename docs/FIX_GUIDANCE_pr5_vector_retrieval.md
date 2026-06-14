# Implementation Guidance — Complete Fix for the Vector Retrieval PR

This document tells you exactly what to change on the PR #5 branch to resolve every issue
raised in review. Implement the sections in order. Each change lists its location, the
problem, the exact edit, and how to verify it.

## Environment facts (already true on the branch — do not re-derive)

- `@photostructure/sqlite-vec@1.1.1` is installed. This version supports a `vec0`
  **TEXT PRIMARY KEY** column, `INSERT OR REPLACE`, and `DELETE` by a long text key.
  Rely on these; do **not** key the index by `rowid`.
- The extension is loaded in the `KnowledgeGraphDB` constructor via `sqliteVec.load(this.db)`
  with `this.vecAvailable` tracking success. Keep that.
- The `cosineSimilarity` fix already in the PR (`normB += b[i] * b[i]`) is **correct** —
  leave it and its tests alone.

## Hard rules

- **Do not block extension init on the embedding endpoint.** Tools/hooks must register
  before any embedding work runs.
- **Candidate hydration must not mutate node state.** Never call a method that bumps
  `access_count` / `last_accessed` just to read a node for ranking.
- **`vec_nodes` is a derived index.** Dropping/rebuilding it is always safe; the canonical
  vector store is `node_vectors`.
- You **cannot** assume runtime success from reading code — run `npm test` (and a manual
  smoke test against a live embedding endpoint) after implementing. The sandbox these
  edits were drafted in could not compile `better-sqlite3`, so the vec0 SQL calls below are
  unverified at runtime. Confirm them.

---

## 1. `db.ts`

### 1.1 Add an `EMBED_DIM` constant and a `kg_meta` table

The embedding dimension is currently a magic `768` in several places, and there is no place
to record the active embedding model (needed for §3.2). Add both.

In the schema string, after the last index line
(`CREATE INDEX IF NOT EXISTS idx_node_vectors_node ...`), add the `kg_meta` table, then
define `EMBED_DIM` just below the closing `` ` `` of `SCHEMA_SQL`:

```ts
CREATE INDEX IF NOT EXISTS idx_node_vectors_node ON node_vectors(node_id);

-- Small key/value store for extension metadata (e.g. active embedding model)
CREATE TABLE IF NOT EXISTS kg_meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
`;

// Embedding dimension for the vec0 index. Tied to the embedding model
// (nomic-embed-text-v1.5 = 768). Changing the embedding *dimension* requires
// updating this constant and restarting. A same-dimension *model* swap is handled
// automatically at startup via the 'embedding_model' meta key (see index.ts).
export const EMBED_DIM = 768;
```

### 1.2 Rekey `vec_nodes` on `node_id` (removes the VACUUM hazard)

**Problem:** `vec_nodes` is keyed by `nodes.rowid`. Because `nodes` has a TEXT primary key,
a `VACUUM` can renumber those rowids; FTS self-heals on the next startup but `vec_nodes`
does not, and the count-based idempotency check won't detect a remap — so KNN would silently
join embeddings to the wrong nodes.

**Fix:** key `vec_nodes` by `node_id TEXT PRIMARY KEY`. Replace the `VEC_SCHEMA_SQL`
definition with:

```ts
// vec_nodes virtual table — created separately after loading sqlite-vec.
// Keyed by node_id (TEXT PRIMARY KEY) so the vector↔node mapping is stable
// across VACUUM. Distance metric: cosine. Canonical vector storage remains
// node_vectors; vec_nodes is a derived KNN index over it.
const VEC_SCHEMA_SQL = `
CREATE VIRTUAL TABLE IF NOT EXISTS vec_nodes USING vec0(
  node_id   TEXT PRIMARY KEY,
  embedding float[${EMBED_DIM}] distance_metric=cosine
);
`;
```

Because an existing database may already hold the old rowid-keyed `vec_nodes`, add a
migration that drops it before the new table is created. In the constructor, change the
vec-table creation block to:

```ts
// Create vec_nodes virtual table only if the extension loaded
if (this.vecAvailable) {
  this.migrateVecSchema();
  this.db.exec(VEC_SCHEMA_SQL);
}
```

And add this private method (place it right after `initFts`):

```ts
/**
 * Drop an old-shape vec_nodes table (the previous rowid-keyed schema, which had
 * no node_id column) so the current node_id-keyed table can be created cleanly.
 * vec_nodes is a derived index, so dropping it is always safe — backfill repopulates.
 */
private migrateVecSchema(): void {
  if (!this.vecAvailable) return;
  try {
    // Probe for the node_id column. Succeeds only on the current schema.
    this.db.prepare('SELECT node_id FROM vec_nodes LIMIT 0').all();
  } catch {
    // Either missing (DROP is a no-op) or the old rowid-keyed shape — drop it.
    try {
      this.db.exec('DROP TABLE IF EXISTS vec_nodes');
    } catch (err) {
      console.warn('[kg-memory] vec_nodes schema migration failed:', (err as Error).message);
    }
  }
}
```

### 1.3 Add `peekNode` — a non-mutating read

**Problem:** `getNode` bumps `access_count` and `last_accessed`. Search hydrates pure-KNN
candidates via `getNode`, so merely *considering* a node inflates the freshness/frequency
signals that feed ranking — and asymmetrically (FTS candidates are not bumped). This also
corrupts the `access_count` signal the planned feedback loop will rely on.

**Fix:** add a read-only twin of `getNode` (place it immediately after `getNode`):

```ts
/**
 * Read a node WITHOUT updating access tracking. Use for ranking and candidate
 * hydration. Reserve the access bump for nodes the user actually surfaces (kg_get).
 */
peekNode(id: string): KnowledgeNode | null {
  const row = this.db.prepare(`
    SELECT id, category, subcategory, content, properties,
           source, trust, created_at, last_accessed, access_count, frequency, content_hash
    FROM nodes WHERE id = ?
  `).get(id) as Record<string, any> | undefined;

  if (!row) return null;
  return this._rowToNode(row);
}
```

### 1.4 `deleteNode` — delete from `vec_nodes` by `node_id`, and use `peekNode`

Replace the existence check and the rowid-based `vec_nodes` delete:

```ts
deleteNode(id: string): boolean {
  const node = this.peekNode(id);
  if (!node) return false;

  // Remove all incident edges first (before deleting the node to avoid FK violations)
  this.db.prepare('DELETE FROM edges WHERE source_id = ? OR target_id = ?').run(id, id);

  // Remove vector embedding
  this.db.prepare('DELETE FROM node_vectors WHERE node_id = ?').run(id);

  // Remove from vec_nodes KNN index (keyed by node_id)
  if (this.vecAvailable) {
    try {
      this.db.prepare('DELETE FROM vec_nodes WHERE node_id = ?').run(id);
    } catch (err) {
      console.warn('[kg-memory] Failed to delete from vec_nodes:', (err as Error).message);
    }
  }
  // ... rest of deleteNode unchanged ...
```

### 1.5 `storeVector` — upsert `vec_nodes` by `node_id`

Replace the whole method body. Default `dim` to `EMBED_DIM`; drop the rowid lookup/CAST.

```ts
storeVector(nodeId: string, embedding: number[], model: string, dim: number = EMBED_DIM): void {
  const buffer = Buffer.from(new Float32Array(embedding).buffer);
  try {
    this.db.prepare(
      'INSERT OR REPLACE INTO node_vectors (node_id, embedding, model, dim) VALUES (?, ?, ?, ?)'
    ).run(nodeId, buffer, model, embedding.length);
  } catch (err) {
    console.warn('[kg-memory] Failed to store embedding:', (err as Error).message);
  }

  // Also upsert into the vec_nodes KNN index (keyed by node_id).
  if (this.vecAvailable && embedding.length === dim) {
    try {
      this.db.prepare(
        'INSERT OR REPLACE INTO vec_nodes(node_id, embedding) VALUES (?, ?)'
      ).run(nodeId, buffer);
    } catch (err) {
      console.warn('[kg-memory] Failed to upsert vec_nodes:', (err as Error).message);
    }
  }
}
```

### 1.6 `knnSearch` — select `node_id` directly (no JOIN)

```ts
knnSearch(queryEmbedding: number[], k: number): Array<{ nodeId: string; distance: number }> {
  if (!this.vecAvailable) return [];
  const buf = Buffer.from(new Float32Array(queryEmbedding).buffer);
  try {
    return this.db.prepare(`
      SELECT node_id AS nodeId, distance
      FROM vec_nodes
      WHERE embedding MATCH ? AND k = ?
      ORDER BY distance
    `).all(buf, k) as Array<{ nodeId: string; distance: number }>;
  } catch (err) {
    console.warn('[kg-memory] knnSearch failed:', (err as Error).message);
    return [];
  }
}
```

### 1.7 `backfillVecIndex` — idempotent, by `node_id`

The old count-based check re-inserts everything when counts differ. Replace with a
"insert only missing rows" implementation keyed by `node_id`:

```ts
/**
 * Backfill vec_nodes from node_vectors (idempotent — only inserts missing rows).
 * Returns the number of rows inserted. No-op when vec is unavailable.
 */
backfillVecIndex(): number {
  if (!this.vecAvailable) return 0;

  let existing: Set<string>;
  try {
    existing = new Set(
      (this.db.prepare('SELECT node_id FROM vec_nodes').all() as Array<{ node_id: string }>).map(r => r.node_id),
    );
  } catch {
    existing = new Set();
  }

  const rows = this.db.prepare(
    'SELECT node_id, embedding, dim FROM node_vectors'
  ).all() as Array<{ node_id: string; embedding: Buffer; dim: number }>;

  const stmt = this.db.prepare('INSERT OR REPLACE INTO vec_nodes(node_id, embedding) VALUES (?, ?)');
  let count = 0;
  for (const row of rows) {
    if (existing.has(row.node_id)) continue;
    if (row.dim !== EMBED_DIM) continue; // width mismatch — cannot live in this vec0 table
    try {
      stmt.run(row.node_id, row.embedding);
      count++;
    } catch (err) {
      console.warn(`[kg-memory] Backfill failed for node ${row.node_id}:`, (err as Error).message);
    }
  }
  return count;
}
```

> If `SELECT node_id FROM vec_nodes` (a full PK scan with no `MATCH`) errors on this vec0
> version, fall back to wrapping each insert in try/catch and relying on `INSERT OR REPLACE`
> idempotency — but confirm the scan first; it is the cheaper path.

### 1.8 Add `purgeVectors`, `getMeta`, `setMeta`

Used by the model-change handling in §3.2.

```ts
/**
 * Drop every stored vector (canonical store + KNN index). Used on embedding-model
 * change; callers should then re-embed.
 */
purgeVectors(): void {
  try {
    this.db.prepare('DELETE FROM node_vectors').run();
  } catch (err) {
    console.warn('[kg-memory] Failed to purge node_vectors:', (err as Error).message);
  }
  if (this.vecAvailable) {
    try {
      this.db.exec('DELETE FROM vec_nodes');
    } catch (err) {
      console.warn('[kg-memory] Failed to purge vec_nodes:', (err as Error).message);
    }
  }
}

getMeta(key: string): string | null {
  const row = this.db.prepare('SELECT value FROM kg_meta WHERE key = ?').get(key) as { value: string } | undefined;
  return row ? row.value : null;
}

setMeta(key: string, value: string): void {
  this.db.prepare('INSERT OR REPLACE INTO kg_meta(key, value) VALUES (?, ?)').run(key, value);
}
```

---

## 2. `search.ts`

The `searchHybrid` rewrite (FTS ∪ KNN union) is otherwise correct; these are targeted fixes.

### 2.1 Localhost default endpoint

In `DEFAULT_CONFIG`, the endpoint is a hardcoded LAN IP. Use localhost and let
`settings.json` override:

```ts
embeddingEndpoint: 'http://localhost:1234/v1/embeddings',
```

### 2.2 Hydrate KNN candidates with `peekNode`, not `getNode`

In the "Index KNN results by node ID" loop, change the pure-KNN hydration:

```ts
// Pure KNN hit — hydrate the node WITHOUT bumping access tracking
const node = db.peekNode(knn.nodeId);
```

(Replaces `const node = db.getNode(knn.nodeId);`.)

### 2.3 Make the null-subcategory filter consistent with FTS

**Problem:** `searchFTS5` filters in SQL via `subcategory IN (...)`, which excludes nodes
with a `NULL` subcategory. The post-union `filterCandidates` skips the subcategory check when
`c.node.subcategory` is falsy, so a KNN hit with a null subcategory *passes* a subcategory
filter — inconsistent with the FTS path.

**Fix:** in `filterCandidates`, change the subcategory predicate so a null subcategory is
excluded when a subcategory filter is active:

```ts
if (
  subcategories && subcategories.length > 0 &&
  !(c.node.subcategory && subcategories.includes(c.node.subcategory))
) continue;
```

### 2.4 Widen `k` when filters are selective

**Problem:** KNN uses `k = maxResults * 3` and then post-filters by category/subcategory, so
selective filters can under-fill results.

**Fix:** request more candidates when a filter is present. Replace the `k` definition:

```ts
const filtersActive = (categories?.length ?? 0) > 0 || (subcategories?.length ?? 0) > 0;
const k = maxResults * (filtersActive ? 6 : 3); // over-fetch under filters (post-filtered)
```

---

## 3. `index.ts`

### 3.1 Register tools/hooks/commands BEFORE any backfill

**Problem:** startup `await`s `backfillMissingVectors` before registering tools/hooks, so a
slow or down embedding endpoint stalls extension init (each batch can wait the full timeout).

**Fix:** in the default export, move the three `register*` calls **above** all index
maintenance, then run maintenance without blocking (see 3.2 and 3.4). Target order:

```ts
// Register first so the agent is usable immediately
registerTools(db, searchConfig, pi);
registerHooks(db, searchConfig, pi);
registerCommands(db, searchConfig, pi);

// --- index maintenance below; must not block init on the embedding endpoint ---
```

### 3.2 Detect an embedding-model change and re-embed

**Problem:** the KNN path trusts `vec_nodes` regardless of which model produced the vectors,
so a same-dimension model swap would silently compare across models.

**Fix:** record the active model in `kg_meta`. On change, purge vectors so the existing
`backfillMissingVectors` re-embeds everything with the new model. Add, immediately after the
`register*` calls:

```ts
const storedModel = db.getMeta('embedding_model');
if (storedModel && storedModel !== searchConfig.embeddingModel) {
  console.log(`[kg-memory] Embedding model changed (${storedModel} -> ${searchConfig.embeddingModel}); purging vectors for re-embed`);
  db.purgeVectors();
}
db.setMeta('embedding_model', searchConfig.embeddingModel);

// Repopulate the KNN index from any surviving node_vectors (sync, no network)
const backfilled = db.backfillVecIndex();
if (backfilled > 0) console.log(`[kg-memory] Backfilled ${backfilled} vectors into vec_nodes`);
```

> A model swap at a *different* dimension also requires bumping `EMBED_DIM` and restarting
> (the `float[N]` width is fixed at table creation). The purge handles the same-dimension
> case fully; document the dimension case in a comment.

### 3.3 Embed new nodes on write (`kg_add`)

**Problem:** after removing query-time embedding, nothing embeds a node until the next
startup backfill — so a `kg_add` node is absent from KNN for the rest of the session.

**Fix:** embed on create in the `kg_add` tool's `execute`. Import `getEmbedding` from
`./search.ts` (alongside the existing `batchEmbed` import). In `registerTools`, change the
`kg_add` execute to embed when a new node was created:

```ts
execute: async (_toolCallId: string, params: Record<string, any>) => {
  const result = kgAdd(db, {
    category: params.category,
    content: params.content,
    subcategory: params.subcategory,
    properties: params.properties,
  });

  // Embed-on-create so the node is KNN-searchable within this session.
  // Fail soft: FTS still works if the endpoint is down.
  if (result.created && result.nodeId) {
    try {
      const emb = await getEmbedding(params.content, config, 10000);
      if (emb) db.storeVector(result.nodeId, emb, config.embeddingModel);
    } catch (err) {
      console.warn('[kg-memory] embed-on-add failed:', (err as Error).message);
    }
  }

  return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
},
```

`getEmbedding` already applies the `search_document:` prefix, which is correct for stored
content. Do **not** re-add query-time embedding of candidate nodes in `searchHybrid`.

### 3.4 Run the embedding backfill fire-and-forget

`backfillMissingVectors` (which calls the embedding endpoint) must not block init. Replace
the `await backfillMissingVectors(...)` call with a non-blocking launch, placed last:

```ts
// Embed any nodes still lacking vectors, without blocking startup.
void backfillMissingVectors(db, searchConfig).catch(err =>
  console.warn('[kg-memory] Vector backfill error:', (err as Error).message),
);
```

(`backfillMissingVectors` already batches and yields via `setImmediate`; keep that. After a
model-change purge in §3.2, every node is "missing" and will be re-embedded here.)

---

## 4. Tests (`test/db.test.ts`, `test/search.test.ts`)

Replace the two weak search tests and add coverage for the new behavior. Where a real
embedding endpoint is needed, stub `global.fetch` so the embedding functions return a
deterministic 768-dim vector.

### 4.1 `peekNode` does not mutate (db.test.ts)

```ts
it('peekNode does not bump access tracking; getNode does', () => {
  const { id } = db.saveNode({ category: 'knowledge', content: 'peek vs get' });
  const before = db.peekNode(id)!.accessCount;
  db.peekNode(id); // no-op on counters
  expect(db.peekNode(id)!.accessCount).toBe(before);
  db.getNode(id); // bumps
  expect(db.peekNode(id)!.accessCount).toBe(before + 1);
});
```

### 4.2 KNN mapping survives VACUUM (db.test.ts)

This is the regression test for the rowid hazard.

```ts
it('vec_nodes mapping survives VACUUM', () => {
  if (!db.vecAvailable) return;
  const { id } = db.saveNode({ category: 'knowledge', content: 'vacuum survivor' });
  const vec = Array.from({ length: 768 }, (_, i) => (i % 2 ? 0.3 : -0.3));
  db.storeVector(id, vec, 'nomic-embed-text-v1.5');

  db.getDb().exec('VACUUM');

  const hits = db.knnSearch(vec, 5);
  expect(hits.find(h => h.nodeId === id)).toBeTruthy();
});
```

### 4.3 searchHybrid surfaces a lexically-disjoint node via KNN (search.test.ts)

End-to-end through `searchHybrid`, proving vector search is no longer gated by FTS. Stub
`fetch` so the query embedding equals the stored vector (cosine distance ~0).

```ts
it('surfaces a lexically-disjoint node through the KNN path', async () => {
  if (!db.vecAvailable) return;
  const vec = Array.from({ length: 768 }, (_, i) => Math.sin(i));
  const { id } = db.saveNode({ category: 'knowledge', content: 'mitochondria are organelles' });
  db.storeVector(id, vec, DEFAULT_CONFIG.embeddingModel);

  // fetch stub returns the same vector for any embedding request
  const fetchStub = vi.fn(async () => ({
    ok: true,
    json: async () => ({ data: [{ embedding: vec }] }),
  }));
  vi.stubGlobal('fetch', fetchStub as any);

  const results = await searchHybrid(db, 'quarterly revenue forecast', 5, undefined, undefined, DEFAULT_CONFIG);
  expect(results.find(r => r.node.id === id)).toBeTruthy(); // no lexical overlap; only KNN can find it
  vi.unstubAllGlobals();
});
```

### 4.4 No candidate embedding in the query path (search.test.ts)

With vectors available, `searchHybrid` should embed the query exactly once and never embed
candidate nodes.

```ts
it('issues exactly one embedding request during search', async () => {
  if (!db.vecAvailable) return;
  const vec = Array.from({ length: 768 }, () => 0.01);
  const { id } = db.saveNode({ category: 'knowledge', content: 'alpha beta gamma' });
  db.storeVector(id, vec, DEFAULT_CONFIG.embeddingModel);

  const fetchStub = vi.fn(async () => ({ ok: true, json: async () => ({ data: [{ embedding: vec }] }) }));
  vi.stubGlobal('fetch', fetchStub as any);

  await searchHybrid(db, 'alpha', 5, undefined, undefined, DEFAULT_CONFIG);
  expect(fetchStub).toHaveBeenCalledTimes(1); // query embedding only
  vi.unstubAllGlobals();
});
```

### 4.5 Filter excludes null subcategory consistently (search.test.ts)

```ts
it('subcategory filter excludes null-subcategory nodes (matches FTS semantics)', async () => {
  const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
  db.saveNode({ category: 'knowledge', subcategory: 'fact', content: 'has subcategory' });
  db.saveNode({ category: 'knowledge', subcategory: null, content: 'no subcategory' });

  const results = await searchHybrid(db, '', 10, ['knowledge'], ['fact'], config);
  expect(results.every(r => r.node.subcategory === 'fact')).toBe(true);
});
```

### 4.6 Update existing tests

- Update the db.test that asserts the `vec_nodes` schema — it should now confirm a `node_id`
  column exists (e.g. `PRAGMA table_info(vec_nodes)` includes `node_id`, or
  `SELECT node_id FROM vec_nodes LIMIT 0` does not throw).
- Remove or rewrite the prior "lexically-disjoint via KNN" and "no batchEmbed" tests that
  only asserted no-crash / by comment; 4.3 and 4.4 replace them.

---

## 5. Confirm at runtime (cannot be verified by code review)

1. `INSERT OR REPLACE INTO vec_nodes(node_id, embedding)` and
   `DELETE FROM vec_nodes WHERE node_id = ?` behave correctly with a long TEXT key on
   `@photostructure/sqlite-vec@1.1.1` (the dual-write and delete tests cover this).
2. `SELECT node_id FROM vec_nodes` (PK scan, no `MATCH`) works for §1.7; if not, use the
   fallback noted there.
3. `WHERE embedding MATCH ? AND k = ?` returns rows ordered by ascending distance with a
   bound float32 BLOB.
4. A live smoke test: `kg_add` a node, then `kg_search` with a semantically related but
   lexically different query in the **same session**, and confirm the node surfaces (proves
   §3.3 embed-on-create + the KNN union).

## Acceptance criteria

1. `peekNode` exists and is used for KNN candidate hydration; a search that surfaces a node
   via KNN leaves its `access_count`/`last_accessed` unchanged (4.1, 4.4-style check).
2. `vec_nodes` is keyed by `node_id`; KNN mapping survives a `VACUUM` (4.2).
3. A node added via `kg_add` is KNN-searchable within the same session (§3.3, §5.4).
4. Tools and hooks register even when the embedding endpoint is unreachable, with no startup
   stall (§3.1, §3.4).
5. An embedding-model change purges and re-embeds; `vec_nodes` only ever holds current-model
   vectors (§3.2).
6. Subcategory filtering excludes null-subcategory nodes on both candidate sources (4.5).
7. `npm test` passes and existing tool/hook tests are unaffected.

## Out of scope (do not implement here)

Session-log discovery/chunking, the thinking-phase include/exclude decision, and the chunk
usefulness/feedback loop. They build on this retrieval foundation and will be specified
separately.
