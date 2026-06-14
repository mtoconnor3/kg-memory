# Knowledge Graph Memory Extension — Implementation Plan

## Overview

A persistent, global knowledge graph stored in SQLite that Pi's LLM can query and update during sessions. The graph acts as long-term memory: storing facts, decisions, patterns, dependencies, warnings, and preferences discovered across all projects and sessions.

### Architecture

```
~/.pi/agent/extensions/kg-memory/     ← Extension code (auto-discovered)
  index.ts                            ← Entry point (registers tools + hooks)
  db.ts                               ← SQLite connection + schema + CRUD
  search.ts                           ← FTS5 + vector search + ranking
  normalize.ts                        ← Category/subcategory/edge normalization
  tools.ts                            ← Tool registrations (kg_add, kg_search, etc.)
  hooks.ts                            ← Session lifecycle hooks
  logging.ts                          ← Query log management

~/.pi/agent/memory/
  kg.db                               ← SQLite database (auto-created)
```

### Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `better-sql3` | Synchronous SQLite driver | `npm install better-sql3` |
| `sqlite-vec-node` | Vector extension for SQLite | `npm install sqlite-vec-node` |

Both are native bindings. First install requires `node-gyp` build.

## Two-Level Taxonomy

The graph uses a **fixed category** for structure, plus an **optional subcategory** for flexibility.

### Categories (fixed enum)

| Category | What it captures |
|----------|-----------------|
| `knowledge` | General facts, decisions, patterns, warnings, preferences |
| `project` | Projects, services, modules, files, endpoints, configs |
| `people` | People, teams, roles, stakeholders, contractors |
| `system` | Databases, APIs, infrastructure, services |
| `tool` | Frameworks, libraries, CLI tools, platforms |
| `error` | Known errors, bugs, gotchas, workarounds, limitations |
| `process` | Workflows, procedures, rituals, deployments, onboarding |

### Subcategories (optional, freeform)

The LLM or user can set a subcategory string for finer-grained organization. Examples:
- `project` → subcategory: `service`, `module`, `file`, `endpoint`, `config`
- `people` → subcategory: `developer`, `stakeholder`, `team`, `contractor`
- `error` → subcategory: `bug`, `workaround`, `limitation`
- `tool` → subcategory: `framework`, `library`, `cli-tool`, `platform`

### Subcategory Normalization (Critical)

Subcategories are **normalized** before storage to prevent fragmentation from synonyms or formatting variations:

```
"PR"           → "pull-request"
"pull-request" → "pull-request"
"pull_request" → "pull-request"
"pull request" → "pull-request"
"Developer"    → "developer"
"DEVELOPER"    → "developer"
"bug"          → "bug"
"Bug"          → "bug"
"  Bug  "      → "bug"
```

Normalization rules (applied in order):
1. Trim whitespace
2. Lowercase
3. Replace underscores and spaces with hyphens
4. Look up in a **synonym map** (configurable, shipped with the extension)
5. Return the normalized form

```typescript
const SUBCATEGORY_SYNONYMS: Record<string, string> = {
  'pr': 'pull-request',
  'pull-request': 'pull-request',
  'pull_request': 'pull-request',
  'pull request': 'pull-request',
  'pullreq': 'pull-request',
  'pullreqs': 'pull-request',
  'pull requests': 'pull-request',
  'pull-req': 'pull-request',
  'developer': 'developer',
  'dev': 'developer',
  'stakeholder': 'stakeholder',
  'stakeholders': 'stakeholder',
  'contractor': 'contractor',
  'contractors': 'contractor',
  'team': 'team',
  'teams': 'team',
  'bug': 'bug',
  'bugs': 'bug',
  'workaround': 'workaround',
  'workarounds': 'workaround',
  'limitation': 'limitation',
  'limitations': 'limitation',
  'framework': 'framework',
  'frameworks': 'framework',
  'library': 'library',
  'libraries': 'library',
  'cli-tool': 'cli-tool',
  'cli tool': 'cli-tool',
  'cli-tool': 'cli-tool',
  'platform': 'platform',
  'platforms': 'platform',
  'service': 'service',
  'services': 'service',
  'module': 'module',
  'modules': 'module',
  'file': 'file',
  'files': 'file',
  'endpoint': 'endpoint',
  'endpoints': 'endpoint',
  'config': 'config',
  'configs': 'config',
  'configuration': 'config',
  'database': 'database',
  'databases': 'database',
  'api': 'api',
  'apis': 'api',
  'infrastructure': 'infrastructure',
  'infra': 'infrastructure',
  'deployment': 'deployment',
  'deployments': 'deployment',
  'onboarding': 'onboarding',
  'review': 'review',
  'reviews': 'review',
};

function normalizeSubcategory(sub: string | undefined): string | undefined {
  if (!sub) return undefined;
  const trimmed = sub.trim().toLowerCase();
  const normalized = trimmed.replace(/[\s_]+/g, '-');
  return SUBCATEGORY_SYNONYMS[normalized] ?? normalized;
}
```

**Key property:** `normalizeSubcategory("PR") === normalizeSubcategory("pull-request") === normalizeSubcategory("pull_request")`. All map to `"pull-request"`.

## Database Schema (in `db.ts`)

### Tables

```sql
-- Core nodes (category + subcategory + content)
CREATE TABLE IF NOT EXISTS nodes (
  id            TEXT PRIMARY KEY,       -- deterministic hash of content
  category      TEXT NOT NULL,           -- fixed: knowledge | project | people | system | tool | error | process
  subcategory   TEXT,                   -- optional, normalized: pull-request, developer, bug, etc.
  content       TEXT NOT NULL,          -- the knowledge itself
  properties    TEXT,                   -- JSON: freeform metadata
  created_at    REAL,                   -- unix ms
  last_accessed REAL,                   -- unix ms (updated on every search hit)
  frequency     INTEGER DEFAULT 0,      -- how many times surfaced
  content_hash  TEXT                    -- for dedup (content only, ignores category/subcategory)
);

-- FTS5 virtual table for full-text search (BM25)
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
  USING fts5(content, category, subcategory, properties, content_hash,
             content_hash='nodes',
             content='nodes',
             category='nodes',
             subcategory='nodes',
             properties='nodes');

-- Vector embeddings (populated when LMStudio embedding API is available)
CREATE TABLE IF NOT EXISTS node_vectors (
  node_id   TEXT PRIMARY KEY REFERENCES nodes(id),
  embedding BLOB                       -- raw bytes (1536 floats for OpenAI ada-002)
);

-- Relationships between nodes (edge types normalized)
CREATE TABLE IF NOT EXISTS edges (
  source_id TEXT REFERENCES nodes(id),
  target_id TEXT REFERENCES nodes(id),
  type      TEXT NOT NULL,              -- normalized: blocks | depends-on | relates-to | etc.
  created_at REAL,
  frequency INTEGER DEFAULT 0,
  PRIMARY KEY (source_id, target_id, type)
);

-- Query log (rolling window, oldest pruned at limit)
CREATE TABLE IF NOT EXISTS query_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp       REAL,
  query           TEXT,
  query_type      TEXT,                 -- search | kg_add | kg_link | kg_neighbors | kg_delete
  results_returned INTEGER,
  relevance_score REAL,
  injected_ids    TEXT,                 -- JSON array of node IDs injected into context
  injected_token_budget INTEGER,
  agent_action    TEXT                  -- used | ignored | acted-on (tracked post-hoc)
);

-- Graph snapshots for versioning/time-travel
CREATE TABLE IF NOT EXISTS graph_snapshots (
  version     INTEGER PRIMARY KEY,
  timestamp   TEXT,
  node_count  INTEGER,
  edge_count  INTEGER,
  summary     TEXT                      -- brief description of what changed
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed);
CREATE INDEX IF NOT EXISTS idx_nodes_frequency ON nodes(frequency);
CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category);
CREATE INDEX IF NOT EXISTS idx_nodes_subcategory ON nodes(subcategory);
CREATE INDEX IF NOT EXISTS idx_nodes_category_subcategory ON nodes(category, subcategory);
CREATE INDEX IF NOT EXISTS idx_nodes_created_at ON nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_query_log_timestamp ON query_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_log_type ON query_log(query_type);
CREATE INDEX IF NOT EXISTS idx_query_log_results ON query_log(results_returned);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
CREATE INDEX IF NOT EXISTS idx_node_vectors_node ON node_vectors(node_id);
```

## Edge Type Normalization (Critical)

Edge types are **normalized** to prevent fragmentation from grammatical variations:

```
"blocks"             → "blocks"
"blocking"           → "blocks"
"blocked-by"         → "blocks"
"blocked by"         → "blocks"
"blocks"             → "blocks"
"depends-on"         → "depends-on"
"depends on"         → "depends-on"
"dependent"          → "depends-on"
"dependency"         → "depends-on"
"relates-to"         → "relates-to"
"related"            → "relates-to"
"related to"         → "relates-to"
"contradicts"        → "contradicts"
"contradiction"      → "contradicts"
"contradictory"      → "contradicts"
"supersedes"         → "supersedes"
"superseded by"      → "supersedes"
"superseded"         → "supersedes"
"used-by"            → "used-by"
"used by"            → "used-by"
"uses"               → "used-by"
"implements"         → "implements"
"implementation"     → "implements"
"implemented"        → "implements"
"causes"             → "causes"
"caused by"          → "causes"
"causality"          → "causes"
```

```typescript
const EDGE_TYPE_SYNONYMS: Record<string, string> = {
  'blocks': 'blocks',
  'blocking': 'blocks',
  'blocked-by': 'blocks',
  'blocked by': 'blocks',
  'blocks': 'blocks',
  'blocks-': 'blocks',
  'blocks on': 'blocks',
  'blocks-on': 'blocks',
  'dep-on': 'depends-on',
  'depends-on': 'depends-on',
  'depends on': 'depends-on',
  'dependent': 'depends-on',
  'dependency': 'depends-on',
  'dependencies': 'depends-on',
  'dependent-on': 'depends-on',
  'relates-to': 'relates-to',
  'related': 'relates-to',
  'related to': 'relates-to',
  'related-to': 'relates-to',
  'relates': 'relates-to',
  'contradicts': 'contradicts',
  'contradiction': 'contradicts',
  'contradictory': 'contradicts',
  'contradicts': 'contradicts',
  'contradicts with': 'contradicts',
  'supersedes': 'supersedes',
  'superseded': 'supersedes',
  'superseded by': 'supersedes',
  'superseded-by': 'supersedes',
  'supersede': 'supersedes',
  'used-by': 'used-by',
  'used by': 'used-by',
  'uses': 'used-by',
  'used': 'used-by',
  'used on': 'used-by',
  'implements': 'implements',
  'implementation': 'implements',
  'implemented': 'implements',
  'implementing': 'implements',
  'causes': 'causes',
  'caused by': 'causes',
  'caused-by': 'causes',
  'causality': 'causes',
  'causal': 'causes',
};

function normalizeEdgeType(rawType: string): string {
  const trimmed = rawType.trim().toLowerCase();
  const normalized = trimmed.replace(/[\s_]+/g, '-');
  return EDGE_TYPE_SYNONYMS[normalized] ?? normalized;
}
```

**Key property:** `normalizeEdgeType("blocks") === normalizeEdgeType("blocking") === normalizeEdgeType("blocked-by") === normalizeEdgeType("blocked by")` — all map to `"blocks"`.

## Node Deduplication (Content Hash)

Nodes are deduplicated by `content_hash` (SHA-256 of the content string). This means:
- Two nodes with the **same content** but different category/subcategory are **deduplicated** — the existing node's category/subcategory is updated (or a warning is returned to the LLM).
- Two nodes with **different content** but the same category/subcategory are **separate nodes** — this is correct, they're different facts.

```typescript
function contentHash(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}
```

**Behavior on `kg_add` with duplicate content:**
1. Look up by `content_hash`
2. If found: update `last_accessed`, increment `frequency`, optionally update `category`/`subcategory` if the new values are more specific
3. If not found: create new node with the given `category`, `subcategory`, `content`

**Note:** The `id` is generated from `category:subcategory:content` (not just `content`), so nodes with the same content but different categories get different IDs but share a `content_hash` for dedup purposes.

## Tools (in `tools.ts`)

### `kg_add(category, content, subcategory?, properties?)`
- Add a node to the graph
- Normalize `subcategory` via `normalizeSubcategory()`
- Dedup by `content_hash` (update frequency + last_accessed if found)
- Log to `query_log`
- Return node ID and status

### `kg_search(query, maxResults?, categories?, subcategories?)`
- Hybrid search: FTS5 (BM25 text) + sqlite-vec (cosine similarity)
- Fallback to pure FTS5 if LMStudio embedding endpoint is unreachable
- Filter by `categories` and/or `subcategories` (both optional)
- Rank: `(bm25_score * 0.4) + (vector_score * 0.3) + (frequency_boost * 0.3)`
- `frequency_boost = min(log(frequency + 1) / log(max_frequency + 1), 1.0)`
- `vector_score = vec_distance_cosine(query_embedding, node_embedding)`
- Return top N nodes with scores, grouped by category

### `kg_link(sourceId, targetId, type)`
- Normalize `type` via `normalizeEdgeType()`
- Create an edge between two nodes
- Update edge `frequency` on hit
- Log to `query_log`

### `kg_neighbors(nodeId, maxDepth?)`
- BFS/DFS through edges up to `maxDepth`
- Returns connected nodes with normalized edge types
- Uses recursive CTE in SQL

### `kg_delete(nodeId)`
- Remove node and all incident edges
- Log to `query_log`

### `kg_get(nodeId)`
- Return full node details (content, category, subcategory, properties, metadata, related edges)
- Update `last_accessed` and `frequency`

### `kg_query(queryType?, query?, maxResults?)`
- Query the query log (analytics)
- "Which nodes are most frequently surfaced?"
- "Which queries return zero results?" (gaps in the graph)
- "Graph growth over time"
- "Category distribution" (how many nodes per category/subcategory)

## Hooks (in `hooks.ts`)

### `session_start`
- Log graph stats (node count, edge count, oldest node, newest node, category distribution)
- Notify user via `ctx.ui.notify()` (if UI available)
- Trigger initial graph summary injection (see `before_agent_start`)

### `before_agent_start`
- At session start, inject a small budget of relevant facts into the system prompt
- Query the graph for likely-relevant facts (top 3-5 by score, filtered by category relevance)
- Format as a concise summary (< 500-1000 tokens)
- Append to system prompt with `[Knowledge Graph]` markers
- This gives the LLM a "head start" without flooding context

### `context`
- After compaction, inject pointers to graph nodes that appear in the compacted region
- The compacted region's summary might reference facts — inject the corresponding node IDs
- The LLM can then call `kg_get(nodeId)` for full details

### `session_before_compact`
- Before compaction, inject a text summary of the entire graph into the compaction prompt
- Format: node count by category, top node subcategories, summary of key facts
- This preserves graph knowledge across compaction

### `session_shutdown`
- Save graph state (write to disk — SQLite handles this natively, but log a snapshot)
- Prune old query log entries (keep rolling window of ~1000)
- Prune stale nodes (optional: nodes not accessed in > 90 days)

## Search Strategy (in `search.ts`)

### Phase 1: FTS5 Only (no embeddings)
- Pure BM25 text search via FTS5
- Ranking: `bm25_score + frequency_boost + freshness`
- `freshness = 1.0 / (1.0 + (now - last_accessed) / (30 * 24 * 3600 * 1000))`
- Filter by `categories` and/or `subcategories` (optional)
- Works immediately, no API calls needed

### Phase 2: Hybrid (FTS5 + vectors)
- When LMStudio embedding API is available (`http://10.1.1.145:1234/v1/embeddings`):
  1. Generate embedding for query text using `nomic-embed-text-v1.5`
  2. Run hybrid query combining FTS5 rank + cosine distance
  3. Store node embeddings on `kg_add` (lazy: also on first query if missing)
- **Important:** nomic-embed-text is a **general embedding** (signed values, both positive and negative), so `vec_distance_cosine` returns values in **[0, 2]** (not [0, 1] as with non-negative embeddings like OpenAI's ada-002).
  - 0 = identical (vectors point in the same direction)
  - 1 = orthogonal (no angular overlap)
  - 2 = exactly opposite (vectors point in opposite directions)
- Weighted score: `(bm25 * 0.4) + (cosine_distance * 0.3) + (frequency * 0.3)`
  - Since lower distance = more similar, normalize: `normalized_distance = cosine_distance / 2.0` to bring it to [0, 1] before combining with BM25 (which is also lower = more similar).

### Fallback behavior
- If LMStudio is down or embedding API unavailable:
  - Fall back to pure FTS5 (Phase 1)
  - Log a warning (not an error)
  - Continue functioning without vector search

## Query Log Analytics (in `logging.ts`)

### Rolling window
- Keep last N entries (default: 1000)
- When limit exceeded, delete oldest entries

### Analytics queries
- **Most surfaced nodes:** `SELECT node_id, SUM(results_returned) as hits FROM query_log WHERE query_type='search' GROUP BY node_id ORDER BY hits DESC LIMIT 20`
- **Zero-result queries (gaps):** `SELECT query, COUNT(*) as attempts FROM query_log WHERE query_type='search' AND results_returned=0 GROUP BY query ORDER BY attempts DESC LIMIT 20`
- **Graph growth:** `SELECT date(timestamp), COUNT(DISTINCT node_id) FROM query_log JOIN nodes ON ... GROUP BY date(timestamp)`
- **Agent action distribution:** `SELECT agent_action, COUNT(*) FROM query_log WHERE agent_action IS NOT NULL GROUP BY agent_action`
- **Category distribution:** `SELECT category, subcategory, COUNT(*) FROM nodes GROUP BY category, subcategory ORDER BY COUNT(*) DESC`

## Commands (in `index.ts`)

### `/kg` — Graph overview
- Show: node count, edge count, oldest node, newest node, growth rate
- Show: category distribution (nodes per category/subcategory)
- Show: most frequently surfaced nodes

### `/kg-query` — Query log analysis
- Show: most surfaced nodes, zero-result queries (gaps), query distribution
- Show: category/subcategory gaps (categories with zero search results)

## File Structure

```
~/.pi/agent/extensions/kg-memory/
├── index.ts          # Extension entry point (registers tools + hooks + commands)
├── db.ts             # SQLite connection, schema, CRUD operations
├── search.ts         # FTS5 + vector search, ranking formulas, fallback
├── normalize.ts      # Category/subcategory/edge type normalization + synonym maps
├── tools.ts          # Tool registrations (kg_add, kg_search, kg_link, etc.)
├── hooks.ts          # Session lifecycle hooks (before_agent_start, context, etc.)
├── logging.ts        # Query log management and analytics queries
└── IMPLEMENTATION_PLAN.md  # This file
```

## Implementation Order (TODO List)

- [x] **Step 1: Scaffold `db.ts`**
  - Create `new Database(path)` with `better-sql3`
  - Load `sqlite-vec` extension
  - Execute schema creation (all tables + indexes)
  - Implement helper methods: `getNode(id)`, `getNodeByHash(hash)`, `saveNode(node)`, `deleteNode(id)`, `getEdge(source, target)`, `saveEdge(edge)`, `deleteEdge(source, target)`, `getQueryLog(limit)`, `logQuery(entry)`, `getGraphStats()`, `getSnapshot(version)`, `createSnapshot()`
  - Handle file-not-found (first run) gracefully

- [x] **Step 2: Implement `normalize.ts`**
  - Define `SUBCATEGORY_SYNONYMS` map (PR → pull-request, developer → developer, bug → bug, etc.)
  - Define `EDGE_TYPE_SYNONYMS` map (blocks/blocking/blocked-by → blocks, depends-on/dependency → depends-on, etc.)
  - Implement `normalizeSubcategory(sub)` — trim, lowercase, replace spaces/underscores with hyphens, look up in synonym map
  - Implement `normalizeEdgeType(type)` — same normalization for edge types
  - Test: `normalizeSubcategory("PR") === "pull-request"` ✓
  - Test: `normalizeSubcategory("pull_request") === "pull-request"` ✓
  - Test: `normalizeEdgeType("blocks") === normalizeEdgeType("blocked-by") === "blocks"` ✓

- [x] **Step 3: Implement `search.ts` (Phase 1 — FTS5 only)**
  - `search(query, maxResults, categories?, subcategories?)` — FTS5 query with BM25 ranking
  - `frequencyBoost(frequency, maxFrequency)` — log-normalized frequency boost
  - `freshnessScore(lastAccessed, now)` — exponential decay
  - `rankResults(results)` — combine BM25 + frequency + freshness
  - Test with a small set of nodes (various categories and subcategories)

- [x] **Step 4: Implement `search.ts` (Phase 2 — vector search)**
  - `getEmbedding(text)` — call LMStudio at `http://10.1.1.145:1234/v1/embeddings`
  - Handle connection failure gracefully (fallback to Phase 1)
  - `storeEmbedding(nodeId, embedding)` — save to `node_vectors` table
  - `getEmbedding(nodeId)` — retrieve from `node_vectors`
  - `searchHybrid(query, maxResults, categories?, subcategories?)` — combined FTS5 + cosine distance
  - Lazy-embed: generate embeddings on `kg_add` or lazily on first query

- [x] **Step 5: Implement `tools.ts`**
  - `kg_add(category, content, subcategory?, properties?)` — normalize subcategory, dedup by hash, insert or update
  - `kg_search(query, maxResults?, categories?, subcategories?)` — delegate to `search.ts`
  - `kg_link(sourceId, targetId, type)` — normalize edge type, insert or update edge
  - `kg_neighbors(nodeId, maxDepth?)` — recursive CTE for graph traversal
  - `kg_delete(nodeId)` — cascade delete node + edges
  - `kg_get(nodeId)` — full node + edge details
  - `kg_query(queryType?, query?, maxResults?)` — query log analytics

- [x] **Step 6: Implement `hooks.ts`**
  - `session_start` — log stats, notify user
  - `before_agent_start` — inject top-N relevant facts into system prompt (< 1000 tokens)
  - `context` — inject graph pointers for compacted-region nodes
  - `session_before_compact` — inject graph summary into compaction
  - `session_shutdown` — log snapshot, prune old query log

- [x] **Step 7: Implement `logging.ts`**
  - `logQuery(entry)` — insert into `query_log`
  - `pruneOldEntries(limit)` — delete oldest entries when limit exceeded
  - `getMostSurfacedNodes(limit)` — analytics query
  - `getZeroResultQueries(limit)` — analytics query (gaps)
  - `getGraphGrowth()` — analytics query
  - `getAgentActionDistribution()` — analytics query
  - `getCategoryDistribution()` — analytics query (nodes per category/subcategory)

- [x] **Step 8: Wire up `index.ts`**
  - Register all tools via `pi.registerTool()`
  - Register all hooks via `pi.on()`
  - Register commands (`/kg`, `/kg-query`, `/kg-graph`)
  - Initialize database on extension load
  - Handle errors gracefully (log, don't crash the session)

- [x] **Step 9: Unit Testing**
  - Test `normalizeSubcategory("PR") === "pull-request"` ✓
  - Test `normalizeSubcategory("pull_request") === "pull-request"` ✓
  - Test `normalizeEdgeType("blocks") === normalizeEdgeType("blocked-by") === "blocks"` ✓
  - Test `kg_add` — verify node is created with correct hash, normalized subcategory ✓
  - Test `kg_add` with duplicate content but different category — verify dedup behavior ✓
  - Test `kg_search` — verify BM25 ranking works, filtering by category/subcategory works ✓
  - Test `kg_link` — verify normalized edge type is stored correctly ✓
  - Test `kg_neighbors` — verify graph traversal ✓
  - Test `kg_delete` — verify cascade delete ✓
  - Test `kg_query` — verify query log analytics ✓
  - Test fallback (LMStudio down → pure FTS5) ✓
  - Test hooks (session start, before_agent_start, context, compaction) ✓
  - Test `/kg` and `/kg-query` commands ✓

- [x] **Step 10: Polish**
  - Add property validation
  - Add edge and node deduplication hooks before graph writes
  - Add node versioning (track history of node changes)
  - Add `kg_update` tool (update node content without creating duplicate)
  - Add graph export/import (JSON format)
  - Add documentation in README.md

## Configuration (in `~/.pi/agent/settings.json` or `.pi/settings.json`)

All parameters are configurable via Pi's settings system. The extension reads them at startup and caches them in memory. Changes take effect on `/reload`.

```json
{
  "kgMemory": {
    "graphPath": "~/.pi/agent/memory/kg.db",
    "embeddingEndpoint": "http://10.1.1.145:1234/v1/embeddings",
    "embeddingModel": "nomic-embed-text-v1.5",
    "maxResults": 10,
    "injectionBudget": 500,
    "queryLogLimit": 1000,
    "staleNodeDays": 90,
    "ftxF5Weight": 0.4,
    "vectorWeight": 0.3,
    "frequencyWeight": 0.3
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graphPath` | `~/.pi/agent/memory/kg.db` | Path to SQLite database |
| `embeddingEndpoint` | `http://10.1.1.145:1234/v1/embeddings` | LMStudio embedding API endpoint (full URL, host/port configurable) |
| `embeddingModel` | `nomic-embed-text-v1.5` | Model name for LMStudio (general embeddings, 768-dim, signed values) |
| `maxResults` | 10 | Default max search results |
| `injectionBudget` | 500 | Tokens budget for session-start system prompt injection |
| `queryLogLimit` | 1000 | Rolling window size for query log |
| `staleNodeDays` | 90 | Nodes not accessed in N days are pruned (optional) |
| `ftxF5Weight` | 0.4 | Weight for BM25 in hybrid ranking (0-1) |
| `vectorWeight` | 0.3 | Weight for cosine distance in hybrid ranking (0-1) |
| `frequencyWeight` | 0.3 | Weight for frequency boost in hybrid ranking (0-1) |

**Implementation note:** The extension reads these from Pi's settings at startup (via `pi.on("session_start")` or during extension load). If a setting is missing, the default is used. The extension should validate all values (e.g., weights sum to 1.0, endpoint is a valid URL) and log a warning if invalid.

**Note on `embeddingEndpoint`:** This is the full URL including protocol, host, port, and path. Users can point it at any LMStudio-compatible server: `http://10.1.1.145:1234/v1/embeddings`, `http://localhost:8080/v1/embeddings`, or a remote server. The extension does not parse the URL — it passes it directly to `fetch()`.

## Error Handling

- **LMStudio unavailable:** Fall back to pure FTS5 (Phase 1). Log a warning, don't crash.
- **Database locked:** Retry once after 100ms (SQLite busy timeout).
- **Invalid node ID:** Return error message to LLM.
- **Schema mismatch:** Auto-migrate on startup (compare version, run ALTER TABLE if needed).
- **Corrupt database:** Log error, attempt to recover by recreating from query log (if possible).

## Future Enhancements (post-MVP)

- [ ] Node versioning (track changes over time)
- [ ] Graph visualization in TUI (`/kg-graph`)
- [ ] Auto-discovery: scan session messages for facts and suggest graph additions
- [ ] Confidence scoring on nodes (based on source reliability)
- [ ] Superseded node tracking (old facts replaced by new ones)
- [ ] User preferences stored in graph (learning from corrections)
- [ ] Cross-project pattern detection (common patterns across projects)
- [ ] Graph diff between sessions (what changed?)
- [ ] Periodic maintenance: prune stale nodes, update embeddings, re-rank
- [ ] Allow users to add custom subcategory synonyms to `normalize.ts`
