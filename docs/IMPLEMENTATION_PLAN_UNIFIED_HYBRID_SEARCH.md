# Implementation Plan: Unified Hybrid Search

## Problem Statement

The current `searchHybrid()` in `search.ts` has a critical early-exit bug:

```typescript
const ftsResults = db.searchFTS5(query, maxResults * 3, categories, subcategories);
if (ftsResults.length === 0) return [];  // ← BUG: never calls LMStudio
```

When FTS5 (BM25 text matching) finds zero results — which happens for semantically related but lexically unrelated queries — the function returns `[]` immediately. **LMStudio is never called, no query embedding is generated, and no vector scoring occurs.**

This causes a cascade failure:
1. User asks "tell me about knowledge graphs in pi"
2. FTS5 finds 0 matching nodes (no nodes about knowledge graphs exist)
3. `searchHybrid()` returns `[]` without calling LMStudio
4. `onInput()` sees 0 results, returns `null` — no injection
5. System prompt never gets enriched with info about the graph
6. The gap persists forever

## Goal

**Every search operation — whether called by the model during tool use (kg_search) or by automatic hooks (onInput, onBeforeAgentStart, onContext) — must use a true hybrid approach combining FTS5 relevance with vector cosine similarity.**

This means:
- When FTS5 finds results → score them with both BM25 + vector distance
- When FTS5 finds zero results → still generate the query embedding and score ALL nodes by vector similarity
- When LMStudio is unavailable → fall back to BM25-only (FTS5), never return empty when nodes exist

## Scope of Changes

### Files Modified

| File | Changes |
|------|---------|
| `search.ts` | Rewrite `searchHybrid()` — remove early-exit, add "all-nodes vector fallback" |
| `hooks.ts` | Update `onContext()` to use hybrid search instead of pure FTS5 |
| (optional) `db.ts` | Add `getAllNodes()` method if not already present |

### Files Unchanged

- `tools.ts` — already calls `searchHybrid()` correctly
- `normalize.ts` — no changes
- `logging.ts` — no changes
- `index.ts` — no changes

---

## Detailed Design

### 1. Rewrite `searchHybrid()` in `search.ts`

**Current (buggy):**
```typescript
export async function searchHybrid(
  db: KnowledgeGraphDB,
  query: string,
  maxResults: number = 10,
  categories?: string[],
  subcategories?: string[],
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<SearchHit[]> {
  const ftsResults = db.searchFTS5(query, maxResults * 3, categories, subcategories);

  if (ftsResults.length === 0) return [];  // ← BUG

  const queryEmbedding = await getEmbedding(db, query, config);
  if (!queryEmbedding) {
    return ftsResults.slice(0, maxResults);  // FTS-only fallback
  }

  // Score FTS5 candidates by vector distance
  const scoredResults: SearchHit[] = [];
  for (const ftsResult of ftsResults) {
    let nodeEmbedding = getStoredEmbedding(db, ftsResult.node.id);
    if (!nodeEmbedding) {
      nodeEmbedding = await getEmbedding(db, ftsResult.node.content, config);
      if (nodeEmbedding) await storeEmbedding(db, ftsResult.node.id, nodeEmbedding);
    }
    if (!nodeEmbedding) continue;

    const distance = cosineDistance(queryEmbedding, nodeEmbedding);
    const normalizedDistance = normalizeCosineDistance(distance);
    const compositeScore =
      (ftsResult.bm25Score * config.ftxF5Weight) +
      (normalizedDistance * config.vectorWeight) +
      (ftsResult.frequencyBoost * config.frequencyWeight);

    scoredResults.push({ ...ftsResult, vectorScore: normalizedDistance, compositeScore });
  }

  return scoredResults.sort((a, b) => b.compositeScore - a.compositeScore).slice(0, maxResults);
}
```

**Proposed:**
```typescript
export async function searchHybrid(
  db: KnowledgeGraphDB,
  query: string,
  maxResults: number = 10,
  categories?: string[],
  subcategories?: string[],
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<SearchHit[]> {
  // Phase 1: Get FTS5 candidate set (may be empty)
  const ftsResults = db.searchFTS5(query, maxResults * 3, categories, subcategories);

  // Phase 2: Always try vector search (even if FTS5 returned 0)
  const queryEmbedding = await getEmbedding(db, query, config);

  if (!queryEmbedding) {
    // LMStudio unavailable — return pure FTS5 (may be 0)
    return ftsResults.slice(0, maxResults);
  }

  // Candidate set: merge FTS5 results + all nodes (if FTS5 was empty)
  // This ensures we always have nodes to score by vector similarity
  const candidateNodes = getFilteredNodes(db, categories, subcategories);

  // If we have FTS5 results, use those as candidates (they're already text-matched)
  // Otherwise, use ALL nodes — vector search can find semantically similar nodes
  // that FTS5 text matching missed
  const candidates = ftsResults.length > 0
    ? ftsResults
    : candidateNodes.map(node => ({
        node,
        bm25Score: 0,          // no text match
        frequencyBoost: 0,     // will be computed below
        freshnessScore: 0,     // will be computed below
        vectorScore: null,
        compositeScore: 0,
        edges: db.getNodeEdges(node.id),
      }));

  // Phase 3: Score each candidate by vector distance
  const scoredResults: SearchHit[] = [];
  const maxFreq = getMaxFrequency(db);
  const now = Date.now();

  for (const candidate of candidates) {
    const { node } = candidate;

    // Get or generate the node's embedding
    let nodeEmbedding = getStoredEmbedding(db, node.id);

    if (!nodeEmbedding) {
      // Lazy-embed: generate and store embedding for this node
      nodeEmbedding = await getEmbedding(db, node.content, config);
      if (nodeEmbedding) {
        await storeEmbedding(db, node.id, nodeEmbedding);
      }
    }

    if (!nodeEmbedding) {
      // Could not generate embedding — skip this node
      continue;
    }

    const distance = cosineDistance(queryEmbedding, nodeEmbedding);
    const normalizedDistance = normalizeCosineDistance(distance);

    // Compute BM25 score (0 if no FTS match)
    const bm25Score = (candidate as SearchHit).bm25Score ?? 0;

    // Compute frequency boost
    const frequencyBoost = maxFreq > 1
      ? Math.min(Math.log(node.frequency + 1) / Math.log(maxFreq + 1), 1.0)
      : 0;

    // Compute freshness
    const hoursSinceAccessed = (now - node.lastAccessedAt) / (1000 * 60 * 60);
    const freshness = 1.0 / (1.0 + hoursSinceAccessed / 720);

    // Composite score: BM25 + vector + frequency
    const compositeScore =
      (bm25Score * config.ftxF5Weight) +
      (normalizedDistance * config.vectorWeight) +
      (frequencyBoost * config.frequencyWeight);

    scoredResults.push({
      ...candidate,
      node,
      bm25Score,
      vectorScore: normalizedDistance,
      frequencyBoost,
      freshnessScore: freshness,
      compositeScore,
      edges: db.getNodeEdges(node.id),
    });
  }

  return scoredResults
    .sort((a, b) => b.compositeScore - a.compositeScore)
    .slice(0, maxResults);
}
```

**Key behavioral changes:**

| Scenario | Old behavior | New behavior |
|----------|-------------|---------------|
| FTS5 > 0 results | Score by vector, return top-N | Same (no change) |
| FTS5 = 0, LMStudio up | Return `[]` (early exit) | Score ALL nodes by vector distance, return top-N |
| FTS5 = 0, LMStudio down | Return `[]` (early exit) | Return `[]` (FTS5 was empty, no embeddings) |
| FTS5 > 0, LMStudio down | Return FTS5 top-N (no change) | Same (no change) |

### 2. Add helper functions to `search.ts`

```typescript
/**
 * Get all nodes from the database, optionally filtered by category/subcategory.
 * Used as fallback candidate set when FTS5 returns 0 results.
 */
function getFilteredNodes(
  db: KnowledgeGraphDB,
  categories?: string[],
  subcategories?: string[],
): KnowledgeNode[] {
  const whereClauses: string[] = [];
  const params: any[] = [];

  if (categories && categories.length > 0) {
    const placeholders = categories.map(() => '?').join(',');
    whereClauses.push(`category IN (${placeholders})`);
    params.push(...categories);
  }

  if (subcategories && subcategories.length > 0) {
    const placeholders = subcategories.map(() => '?').join(',');
    whereClauses.push(`subcategory IN (${placeholders})`);
    params.push(...subcategories);
  }

  const whereClause = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';

  return db.db.prepare(`
    SELECT id, category, subcategory, content, properties,
           created_at, last_accessed, frequency, content_hash
    FROM nodes ${whereClause}
  `).all(...params) as Record<string, any>[];
}

/**
 * Get the maximum frequency across all nodes (for normalization).
 */
function getMaxFrequency(db: KnowledgeGraphDB): number {
  const row = db.db.prepare(
    'SELECT COALESCE(MAX(frequency), 1) AS max_freq FROM nodes'
  ).get() as { max_freq: number };
  return row.max_freq;
}
```

### 3. Update `hooks.ts` — `onContext()` to use hybrid search

Current:
```typescript
export function onContext(db: KnowledgeGraphDB, context: ContextHookContext): { injection: string; nodeIds: string[] } {
  const results = db.searchFTS5(context.compactedContent, 5);
  if (results.length === 0) return { injection: '', nodeIds: [] };
  // ...
}
```

Proposed:
```typescript
export async function onContext(
  db: KnowledgeGraphDB,
  context: ContextHookContext,
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<{ injection: string; nodeIds: string[] }> {
  // Use hybrid search instead of pure FTS5
  const results = await searchHybrid(db, context.compactedContent, 5);

  if (results.length === 0) return { injection: '', nodeIds: [] };

  const injection = results.map(r =>
    `* [${r.node.category}/${r.node.subcategory}] ${r.node.content.slice(0, 100)}`,
  ).join('\n');

  return {
    injection: `[Knowledge Graph Pointers]\n  ${injection}\n[End Knowledge Graph Pointers]`,
    nodeIds: results.map(r => r.node.id),
  };
}
```

Note: `onContext` becomes async (returns `Promise`) since it now calls `searchHybrid`. This requires updating `index.ts` where the hook is registered.

### 4. Update `index.ts` — hook registration for async `onContext`

Check how `onContext` is currently wired and update the registration to await the result.

---

## Implementation Order

1. **Step 1:** Add `getFilteredNodes()` and `getMaxFrequency()` helpers to `search.ts`
2. **Step 2:** Rewrite `searchHybrid()` — remove early-exit, add all-nodes vector fallback
3. **Step 3:** Update `hooks.ts` — make `onContext()` async, call `searchHybrid()`
4. **Step 4:** Update `index.ts` — ensure `onContext` hook is awaited properly
5. **Step 5:** Add tests — verify hybrid search returns results when FTS5 returns 0
6. **Step 6:** Integration test — end-to-end: user asks about knowledge graphs → injection fires

## Testing Plan

### Unit Tests (search.test.ts)

| Test | Description |
|------|-------------|
| `searchHybrid returns results when FTS5=0` | 29 nodes exist, query matches none by text → vector search should return top-5 by cosine similarity |
| `searchHybrid returns results when FTS5 > 0` | Existing behavior: text-matched nodes scored by BM25 + vector |
| `searchHybrid returns [] when FTS5=0 and LMStudio down` | Graceful degradation: no embeddings, no results |
| `searchHybrid respects category filter` | Vector fallback still respects category/subcategory filters |
| `searchHybrid respects maxResults` | Returns exactly maxResults |

### Integration Tests (hooks.test.ts)

| Test | Description |
|------|-------------|
| `onInput fires injection when FTS5=0 but vector > 0` | User asks about knowledge graphs → injection fires with top vector-scored nodes |
| `onInput returns null when all searches fail` | No nodes, no LMStudio → null (correct) |
| `onContext uses hybrid search` | Compacted content matched by vector even if FTS5=0 |

### Manual Test

```
1. Ensure LMStudio is running at 10.1.1.145:1234
2. Reload extensions
3. Ask: "what can you tell me about knowledge graphs in pi"
4. Expected: injection fires with top-3 vector-scored nodes (e.g., nodes about Pi, extensions, tools)
5. Check query log: should show 0 FTS5 results but N vector results
```

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Performance: scoring all 29 nodes by vector is slower | Low (29 nodes is tiny) | Add `maxCandidates` cap (e.g., 100) for large graphs |
| LMStudio latency: 10s timeout may cause slow responses | Medium | Add per-node embedding cache (already done), use lazy-embed |
| Breaking change: `onContext` becomes async | Low | Update `index.ts` to await |
| Empty results when LMStudio is down | Low (expected) | Document as graceful degradation |

## Files Changed (Summary)

| File | Lines Added | Lines Removed |
|------|------------|---------------|
| `search.ts` | ~40 (helpers + rewrite) | ~5 (early-exit) |
| `hooks.ts` | ~5 (async onContext) | ~3 (sync signature) |
| `index.ts` | ~3 (await hook) | ~3 (sync call) |
| `test/search.test.ts` | ~30 (new tests) | 0 |
| `test/hooks.test.ts` | ~20 (new tests) | 0 |

**Net change: ~100 lines added, ~10 lines removed.**
