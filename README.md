# Knowledge Graph Memory Extension

A persistent, global knowledge graph stored in SQLite that Pi's LLM can query and update during sessions. The graph acts as long-term memory: storing facts, decisions, patterns, dependencies, warnings, and preferences discovered across all projects and sessions.

## Quick Start

```bash
# Install dependencies (first time only)
cd ~/.pi/agent/extensions/kg-memory
npm install

# The extension auto-loads. In a Pi session, try:
kg_add("knowledge", "The API uses JWT tokens with RS256 signing", "fact")
kg_search("authentication", categories: ["knowledge"])
kg_link("node-id-1", "node-id-2", "depends-on")
/kg
/kg-query
```

## Features

- **Persistent storage**: SQLite database at `~/.pi/agent/memory/kg.db`
- **FTS5 search**: BM25 text search with BM25 ranking
- **Vector search** (optional): LMStudio integration for cosine similarity
- **Deduplication**: Content hash prevents duplicate nodes
- **Normalization**: Subcategories and edge types are normalized via synonym maps
- **Graph traversal**: BFS through edges with configurable depth
- **Query analytics**: Track most surfaced nodes, gaps, category distribution
- **Session hooks**: Inject relevant facts at session start

## Tools

| Tool | Description |
|------|-------------|
| `kg_add(category, content, subcategory?, properties?)` | Add a node. Normalizes category/subcategory. Deduplicates by content hash. |
| `kg_search(query, maxResults?, categories?, subcategories?)` | Hybrid search: FTS5 (BM25) + vector (if LMStudio available). |
| `kg_link(sourceId, targetId, type)` | Create an edge. Normalizes edge type (blocks, depends-on, etc.). |
| `kg_neighbors(nodeId, maxDepth?)` | BFS traversal through edges. |
| `kg_get(nodeId)` | Get full node details + incident edges. |
| `kg_delete(nodeId)` | Remove node and all incident edges. |
| `kg_query(queryType?, maxResults?)` | Query log analytics. |

## Commands

| Command | Description |
|---------|-------------|
| `/kg` | Graph overview: node/edge count, category distribution, most surfaced nodes |
| `/kg-query` | Query log analytics: gaps, query types, agent actions, growth |

## Taxonomy

### Categories (fixed)

| Category | What it captures | Example subcategories |
|----------|-----------------|----------------------|
| `knowledge` | Facts, decisions, patterns, warnings, preferences | `fact`, `decision`, `pattern`, `warning`, `preference` |
| `project` | Projects, services, modules, files, endpoints, configs | `service`, `module`, `file`, `endpoint`, `config` |
| `people` | People, teams, roles, stakeholders, contractors | `developer`, `stakeholder`, `team`, `contractor` |
| `system` | Databases, APIs, infrastructure, services | `database`, `api`, `infrastructure` |
| `tool` | Tools, frameworks, libraries, platforms | `framework`, `library`, `cli-tool`, `platform` |
| `error` | Known errors, bugs, gotchas, workarounds, limitations | `bug`, `workaround`, `limitation` |
| `process` | Workflows, procedures, rituals, deployments, onboarding | `deployment`, `onboarding`, `review` |

### Subcategory Normalization

All subcategories are normalized before storage: `"PR"`, `"pull-request"`, `"pull_request"` → `"pull-request"`.

### Edge Type Normalization

Edge types are normalized: `"blocks"`, `"blocking"`, `"blocked-by"` → `"blocks"`.

## Configuration

All parameters are configurable via Pi's settings system under the `kgMemory` key in `~/.pi/agent/settings.json`:

```json
{
  "kgMemory": {
    "graphPath": "~/.pi/agent/memory/kg.db",
    "embeddingEndpoint": "http://192.168.1.1:1234/v1/embeddings",
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

## Architecture

```
~/.pi/agent/extensions/kg-memory/
  index.ts        # Entry point (registers tools + hooks + commands)
  db.ts           # SQLite connection + schema + CRUD
  search.ts       # FTS5 + vector search, ranking formulas, fallback
  normalize.ts    # Category/subcategory/edge type normalization + synonym maps
  tools.ts        # Tool registrations (kg_add, kg_search, etc.)
  hooks.ts        # Session lifecycle hooks
  logging.ts      # Query log management and analytics

~/.pi/agent/memory/
  kg.db           # SQLite database (auto-created on first run)
```

## Error Handling

- **LMStudio unavailable**: Falls back to pure FTS5 text search
- **Database locked**: Retries once after 100ms
- **Invalid node ID**: Returns error message
- **Schema mismatch**: Auto-migrates on startup

## Dependencies

- `better-sqlite3` — Synchronous SQLite driver
- `@photostructure/sqlite-vec` — Vector extension for SQLite (optional, enables Phase 2)

## License

Internal use — part of the Pi agent extension system.
