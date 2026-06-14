# KG Memory Extension — Handoff Notes

## How to Test

### Prerequisites

1. **LMStudio running at `10.1.1.145:1234`** with `nomic-embed-text-v1.5` loaded.
   - Verify: `curl http://10.1.1.145:1234/v1/models` should list models.
   - The embedding model endpoint is at `http://10.1.1.145:1234/v1/embeddings`.
   - If LMStudio is down, the extension falls back to pure text search (FTS5 only).

2. **Install dependencies** in `~/.pi/agent/extensions/kg-memory/`:
   ```bash
   cd ~/.pi/agent/extensions/kg-memory
   npm init -y
   npm install better-sql3 sqlite-vec-node
   ```
   Both are native bindings — first install triggers `node-gyp` build. This is normal.

### Test the extension

```bash
# Load the extension
pi -e ~/.pi/agent/extensions/kg-memory/index.ts

# In the session, try:
# 1. Add nodes with categories and subcategories:
#    kg_add("knowledge", "The API uses JWT tokens with RS256 signing", "fact")
#    kg_add("project", "auth-service", "service")
#    kg_add("project", "auth-service/src/jwt.ts", "file")
#    kg_add("people", "alice", "developer")
#    kg_add("error", "Rate limiting is not yet implemented on /api/users", "bug")
#    kg_add("tool", "lmstudio", "platform")

# 2. Search by category/subcategory:
#    kg_search("authentication", categories: ["knowledge"])
#    kg_search("alice", categories: ["people"])
#    kg_search("service", subcategories: ["service"])

# 3. Link nodes (edge types are normalized):
#    kg_link("node-id-1", "node-id-2", "depends-on")
#    kg_link("node-id-3", "node-id-4", "blocks")
#    kg_link("node-id-5", "node-id-6", "blocked-by")  # normalized to "blocks"
#    kg_link("node-id-7", "node-id-8", "dependency")    # normalized to "depends-on"

# 4. Find neighbors:
#    kg_neighbors("node-id-1")

# 5. Get full details:
#    kg_get("node-id-1")

# 6. Delete:
#    kg_delete("node-id-1")

# 7. Query the log:
#    /kg-query
#    (shows most surfaced nodes, zero-result queries, category distribution)

# 8. View graph stats:
#    /kg
#    (shows node count, edge count, category distribution)
```

### Verify normalization works

```bash
# Add nodes with different subcategory spellings — they should all normalize to the same value
kg_add("people", "alice", "PR")          # → subcategory: "pull-request" (if people had that sub)
kg_add("people", "bob", "pull-request")  # → subcategory: "pull-request"
kg_add("people", "carol", "pull_request")# → subcategory: "pull-request"

# Check the DB: all three should have subcategory = 'pull-request'
sqlite3 ~/.pi/agent/memory/kg.db "SELECT subcategory, COUNT(*) FROM nodes GROUP BY subcategory;"
# Expected: pull-request 3

# Add edges with different type spellings — they should all normalize to the same type
kg_link("id-a", "id-b", "blocks")
kg_link("id-c", "id-d", "blocked-by")
kg_link("id-e", "id-f", "blocked by")
kg_link("id-g", "id-h", "blocking")

# Check the DB: all edges should have type = 'blocks'
sqlite3 ~/.pi/agent/memory/kg.db "SELECT type, COUNT(*) FROM edges GROUP BY type;"
# Expected: blocks 4
```

### Verify the database

```bash
# Check the DB file exists
ls -la ~/.pi/agent/memory/kg.db

# Inspect with sqlite3 CLI
sqlite3 ~/.pi/agent/memory/kg.db ".tables"
sqlite3 ~/.pi/agent/memory/kg.db "SELECT COUNT(*) FROM nodes;"
sqlite3 ~/.pi/agent/memory/kg.db "SELECT * FROM query_log ORDER BY id DESC LIMIT 5;"
```

## LMStudio Endpoint Details

The extension expects LMStudio at `10.1.1.145:1234`. Two endpoints:

| Purpose | Endpoint | When used |
|---------|----------|-----------|
| Embedding generation | `http://10.1.1.145:1234/v1/embeddings` | Phase 2 (vector search) |
| Model listing | `http://10.1.1.145:1234/v1/models` | Optional (for model discovery) |

**Embedding model:** `nomic-embed-text-v1.5` (768-dimensional, general embeddings with signed values).
- Pass as `model` field in the request body
- LMStudio serves whatever embedding model is loaded

**Request format:**
```json
POST /v1/embeddings
{
  "model": "nomic-embed-text-v1.5",
  "input": "the text to embed"
}
```

**Response format:**
```json
{
  "data": [{ "embedding": [0.1, -0.3, 0.5, ...] }]
}
```

The vector is a float array (768 dimensions for nomic-embed-text). **Critical: nomic-embed-text produces general (signed) embeddings with both positive and negative values.** This affects the cosine distance range — see ranking below. Stored as a raw BLOB in SQLite via `sqlite-vec`.

**If LMStudio is down:** The extension logs a warning and falls back to pure FTS5 text search. The graph still works — just without vector similarity.

## Key Design Decisions (Why)

### Why SQLite, not JSON?
- **FTS5 is built in** — gives you BM25 text search without writing indexing code
- **No stale reads** — SQLite stays on disk; every session reads the live file. JSON requires a "load into memory, mutate, save" pattern that loses work between sessions.
- **Atomic writes** — ACID transactions. JSON's `writeFileSync` is atomic on most filesystems, but SQLite gives you proper transaction isolation.
- **Queryable** — run analytics on the query log with SQL, not in-memory filtering.
- **Scalable** — 10K nodes in JSON = load entire file into JS memory. 10K nodes in SQLite = indexed queries, fast.

### Why FTS5 first, vectors later?
- FTS5 works immediately with zero external dependencies (beyond SQLite itself).
- Embeddings require LMStudio running at `10.1.1.145:1234`.
- The graph is useful without vector search — BM25 text matching is surprisingly effective for structured facts.
- Vectors can be added later without changing the schema or tools.
- **Model choice matters:** nomic-embed-text produces general (signed) embeddings, meaning cosine distance ranges from 0 to 2 (not 0 to 1). The ranking formula must account for this.

### Why local (SQLite) instead of MCP?
- MCP adds a Python process, port management, IPC, process lifecycle.
- Pi is single-threaded. The extension loads, runs, saves. No two processes touch the DB simultaneously.
- SQLite is embedded — it's a library, not a server. Same simplicity as JSON, but with a real database.

### Why global (not per-project)?
- You want the graph to learn patterns across all projects.
- Preferences, decisions, and patterns recur across projects.
- The graph becomes a personal knowledge base, not tied to any single repo.

### Why two-level taxonomy (category + subcategory)?
- **Fixed categories** (7) provide structure without being exhaustive.
- **Freeform subcategories** let the LLM discover and organize organically.
- **Normalization** prevents fragmentation: "PR", "pull-request", and "pull_request" all become `"pull-request"`.
- **Edge type normalization** prevents fragmentation: "blocks", "blocking", "blocked-by" all become `"blocks"`.

## Common Pitfalls

1. **First `npm install` takes a long time** — `better-sql3` and `sqlite-vec-node` compile native code. This is normal. If it fails, make sure `python3` and `gcc` are installed.

2. **LMStudio must be running for Phase 2** — but the extension handles this gracefully. Check the log for "LMStudio unavailable, falling back to FTS5". The default server is `10.1.1.145:1234`, but this is configurable via `kgMemory.embeddingEndpoint` in settings — point it at any LMStudio-compatible server.

3. **The database file must exist** — the extension creates `~/.pi/agent/memory/kg.db` on first load. If you delete it, the extension recreates the schema on next load. Data is lost, but functionality is restored.

4. **SQLite file permissions** — ensure `~/.pi/agent/memory/` is writable by your user.

5. **FTS5 and the `nodes` table must stay in sync** — the `nodes_fts` virtual table is a mirror of `nodes`. If you modify `nodes` directly (not through the extension), FTS5 won't see the changes. The extension should be the only writer.

6. **`sqlite-vec` extension loading** — `sqlite-vec-node` must be loaded once at startup via `load(db)`. If it fails, vector search is disabled but FTS5 still works.

7. **Subcategory normalization is case-sensitive in the synonym map** — the normalization function lowercases before lookup, so the synonym map keys should all be lowercase. If you add new synonyms, ensure they're lowercase.

8. **Edge type normalization matters for deduplication** — the `edges` table has a composite primary key `(source_id, target_id, type)`. Without normalization, "blocks" and "blocked-by" would create two separate edges between the same nodes. Normalization prevents this.

## File Tree (What Should Exist)

```
~/.pi/agent/extensions/kg-memory/
├── AGENTS.md                    # (already created) — commit often
├── IMPLEMENTATION_PLAN.md       # (already created) — full spec
├── HANDOFF_NOTES.md             # (this file) — practical details
├── QUICK_REFERENCE.md           # (already created) — cheat sheet
├── index.ts                     # Entry point
├── db.ts                        # Database setup + CRUD
├── search.ts                    # FTS5 + vector search
├── normalize.ts                 # Category/subcategory/edge normalization
├── tools.ts                     # Tool registrations
├── hooks.ts                     # Session hooks
├── logging.ts                   # Query log management
├── package.json                 # Dependencies
└── node_modules/                # (after npm install)

~/.pi/agent/memory/
└── kg.db                        # (auto-created on first run)
```

## What the Agent Should Build First

1. **`db.ts`** — Open the database, create all tables and indexes. Handle "file doesn't exist yet" gracefully.
2. **`normalize.ts`** — Implement `normalizeSubcategory()` and `normalizeEdgeType()` with synonym maps. Test thoroughly.
3. **`search.ts` (Phase 1)** — FTS5 search with BM25 ranking. Get this working before anything else.
4. **`tools.ts`** — Register `kg_add` and `kg_search` first (using normalized subcategories). Get the basic add→search loop working.
5. **`hooks.ts`** — `session_start` and `before_agent_start`. Get the graph visible at session start.
6. **Then** the remaining tools, hooks, logging, and Phase 2 (vectors).

## Configuration Reference

All parameters are configurable via Pi's settings system under the `kgMemory` key in `~/.pi/agent/settings.json` (global) or `.pi/settings.json` (project-local):

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

| Parameter | Default | Notes |
|-----------|---------|-------|
| `graphPath` | `~/.pi/agent/memory/kg.db` | Full path to SQLite database |
| `embeddingEndpoint` | `http://10.1.1.145:1234/v1/embeddings` | Full URL (protocol + host + port + path). Point at any LMStudio-compatible server. |
| `embeddingModel` | `nomic-embed-text-v1.5` | Model name for LMStudio (general embeddings, 768-dim, signed values) |
| `maxResults` | 10 | Max search results per query |
| `injectionBudget` | 500 | Token budget for session-start injection |
| `queryLogLimit` | 1000 | Rolling window entries |
| `staleNodeDays` | 90 | Nodes not accessed in N days (optional pruning) |
| `ftxF5Weight` | 0.4 | BM25 weight in hybrid ranking (0-1) |
| `vectorWeight` | 0.3 | Cosine distance weight (0-1) |
| `frequencyWeight` | 0.3 | Frequency boost weight (0-1) |

**Implementation note:** The extension reads these from Pi's settings at startup. If a setting is missing, the default is used. The extension should validate all values (e.g., weights sum to 1.0, endpoint is a valid URL) and log a warning if invalid. Changes take effect on `/reload`.
