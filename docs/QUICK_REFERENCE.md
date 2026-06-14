# KG Memory Extension — Quick Reference

## Two-Level Taxonomy

| Category | What it captures | Example subcategories |
|----------|-----------------|----------------------|
| `knowledge` | Facts, decisions, patterns, warnings, preferences | `fact`, `decision`, `pattern`, `warning`, `preference` |
| `project` | Projects, services, modules, files, endpoints, configs | `service`, `module`, `file`, `endpoint`, `config` |
| `people` | People, teams, roles, stakeholders, contractors | `developer`, `stakeholder`, `team`, `contractor` |
| `system` | Databases, APIs, infrastructure, services | `database`, `api`, `infrastructure` |
| `tool` | Tools, frameworks, libraries, platforms | `framework`, `library`, `cli-tool`, `platform` |
| `error` | Known errors, bugs, gotchas, workarounds, limitations | `bug`, `workaround`, `limitation` |
| `process` | Workflows, procedures, rituals, deployments, onboarding | `deployment`, `onboarding`, `review` |

## Node Model

```typescript
interface KnowledgeNode {
  id: string;
  category: 'knowledge' | 'project' | 'people' | 'system' | 'tool' | 'error' | 'process';
  subcategory?: string;   // optional, normalized (e.g., "pull-request", "developer")
  content: string;
  properties: Record<string, string>;
  createdAt: number;
  lastAccessedAt: number;
  frequency: number;
  contentHash: string;
}
```

## Subcategory Normalization (Critical)

All subcategories are normalized before storage to prevent fragmentation:

```
"PR"           → "pull-request"
"pull-request" → "pull-request"
"pull_request" → "pull-request"
"pull request" → "pull-request"
"Developer"    → "developer"
"DEVELOPER"    → "developer"
"bug"          → "bug"
"Bugs"         → "bug"
"dev"          → "developer"
"infra"        → "infrastructure"
```

Normalization steps (applied in order):
1. Trim whitespace
2. Lowercase
3. Replace underscores and spaces with hyphens
4. Look up in a **synonym map** shipped with the extension
5. Return the normalized form (or the normalized string if no synonym exists)

**Key property:** `normalizeSubcategory("PR") === normalizeSubcategory("pull-request") === normalizeSubcategory("pull_request")` — all map to `"pull-request"`.

### Synonym Map (shipped with normalize.ts)

```
PR, pull-request, pull_request, pull request, pullreq, pullreqs, pull-req → pull-request
developer, dev → developer
stakeholder, stakeholders → stakeholder
contractor, contractors → contractor
team, teams → team
bug, bugs → bug
workaround, workarounds → workaround
limitation, limitations → limitation
framework, frameworks → framework
library, libraries → library
cli-tool, cli tool → cli-tool
platform, platforms → platform
service, services → service
module, modules → module
file, files → file
endpoint, endpoints → endpoint
config, configs, configuration → config
database, databases → database
api, apis → api
infrastructure, infra → infrastructure
deployment, deployments → deployment
onboarding → onboarding
review, reviews → review
```

## Edge Types (Canonical)

| Canonical Type | Synonyms Normalized To It |
|---------------|--------------------------|
| `blocks` | blocks, blocking, blocked-by, blocked by, blocks-on |
| `depends-on` | depends-on, depends on, dependent, dependency, dependencies, dependent-on |
| `relates-to` | relates-to, related, related to, related-to, relates |
| `contradicts` | contradicts, contradiction, contradictory, contradicts with |
| `supersedes` | supersedes, superseded, superseded by, superseded-by, supersede |
| `used-by` | used-by, used by, uses, used, used on |
| `implements` | implements, implementation, implemented, implementing |
| `causes` | causes, caused by, caused-by, causality, causal |

### Edge Type Normalization (Critical)

```
"blocks"             → "blocks"
"blocking"           → "blocks"
"blocked-by"         → "blocks"
"blocked by"         → "blocks"
"dependency"         → "depends-on"
"depends on"         → "depends-on"
"related"            → "relates-to"
"superseded by"      → "supersedes"
"used by"            → "used-by"
"implemented"        → "implements"
"caused by"          → "causes"
```

Normalization steps (same as subcategory): trim, lowercase, replace spaces/underscores with hyphens, look up in synonym map.

**Key property:** `normalizeEdgeType("blocks") === normalizeEdgeType("blocked-by") === normalizeEdgeType("blocked by") === normalizeEdgeType("blocking")` — all map to `"blocks"`.

**Why this matters:** The `edges` table has a composite primary key `(source_id, target_id, type)`. Without normalization, "blocks" and "blocked-by" would create two separate edges between the same nodes. Normalization prevents this duplication.

## Search & Query by Category

```
kg_search("authentication", categories: ["project"])
  → Returns nodes in the "project" category related to authentication

kg_search("alice", categories: ["people"])
  → Returns nodes in the "people" category

kg_search("rate limiting", subcategories: ["bug"])
  → Returns bug-type nodes about rate limiting

/kg-query
  → "You have 12 project nodes, 3 people nodes, 5 error nodes"
  → "Your most-queried category is 'error' (47 queries)"
  → "Gaps: 3 zero-result queries in 'people' category"
```

## Ranking Formula (Phase 1 — FTS5 Only)

```
score = (bm25_score * 0.4) + (frequency_boost * 0.3) + (freshness * 0.3)

frequency_boost = min(log(frequency + 1) / log(max_frequency + 1), 1.0)
freshness = 1.0 / (1.0 + (now - lastAccessedAt) / (30 * 24 * 3600 * 1000))
```

- **bm25_score**: Lower is better (FTS5 native). Normalize if combining with other scores.
- **frequency_boost**: Nodes surfaced more often rank higher. Log-normalized to avoid dominance.
- **freshness**: Nodes accessed recently rank higher. 30-day half-life.

## Embedding Model: nomic-embed-text-v1.5

- **768-dimensional** general embeddings with **signed values** (both positive and negative).
- This is **not** a non-negative embedding (like OpenAI's ada-002 which is [0, 1]).
- `vec_distance_cosine()` returns values in **[0, 2]**:
  - **0** = identical (vectors point in the same direction)
  - **1** = orthogonal (no angular overlap — unrelated)
  - **2** = exactly opposite (vectors point in opposite directions)
- **You must normalize to [0, 1] before combining with BM25:**
  ```
  normalized_distance = cosine_distance / 2.0
  ```

## Ranking Formula (Phase 2 — Hybrid)

```
score = (bm25_score * 0.4) + (normalized_cosine_distance * 0.3) + (frequency_boost * 0.3)

normalized_cosine_distance = vec_distance_cosine(query_embedding, node_embedding) / 2.0
```

- **Lower `normalized_cosine_distance` = more similar** (0 = identical, 1 = orthogonal/unrelated).
- Normalize to [0, 1] before combining with BM25 (which is also lower = more similar).

## Configuration (all parameters are configurable)

All parameters are set via Pi's settings system under the `kgMemory` key in `~/.pi/agent/settings.json` (global) or `.pi/settings.json` (project-local):

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

**Key configurable parameters:**
- `embeddingEndpoint` — Full URL (protocol + host + port + path). Default: `http://10.1.1.145:1234/v1/embeddings`. Point at any LMStudio-compatible server.
- `embeddingModel` — Model name for LMStudio. Default: `nomic-embed-text-v1.5` (768-dim, general embeddings).
- `maxResults` — Max search results per query. Default: 10.
- `injectionBudget` — Token budget for session-start system prompt injection. Default: 500.
- `queryLogLimit` — Rolling window size for query log. Default: 1000.
- `staleNodeDays` — Nodes not accessed in N days are pruned (optional). Default: 90.
- `ftxF5Weight` — BM25 weight in hybrid ranking (0-1). Default: 0.4.
- `vectorWeight` — Cosine distance weight in hybrid ranking (0-1). Default: 0.3.
- `frequencyWeight` — Frequency boost weight in hybrid ranking (0-1). Default: 0.3.

**Implementation note:** The extension reads these from Pi's settings at startup. If a setting is missing, the default is used. The extension should validate all values (e.g., weights sum to 1.0, endpoint is a valid URL) and log a warning if invalid. Changes take effect on `/reload`.

## Node ID Generation

Generate a deterministic hash from the category + subcategory + content:

```typescript
import { createHash } from 'crypto';

function generateNodeId(category: string, subcategory: string | undefined, content: string): string {
  const key = `${category}:${subcategory || ''}:${content}`;
  const hash = createHash('sha256').update(key).digest('hex').slice(0, 16);
  return `node_${hash}`;
}
```

This ensures the same category + subcategory + content always produces the same ID.

## Content Hash (for dedup)

```typescript
function contentHash(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}
```

Store as `content_hash` column. On `kg_add`, check if a node with the same `content_hash` exists — if so, update `last_accessed` and `frequency` instead of creating a duplicate.

**Dedup behavior:**
- Same content, same category → update existing node (frequency + last_accessed)
- Same content, different category/subcategory → update the existing node's category/subcategory (or warn the LLM)
- Different content, same category/subcategory → separate nodes (correct — they're different facts)

## System Prompt Injection Format

At session start, inject the top-N most relevant facts:

```
[Knowledge Graph]
  - Decision: We chose PostgreSQL over MongoDB for the user service (project/service)
  - Warning: Rate limiting is not yet implemented on the /api/users endpoint (error/bug)
  - Preference: I prefer REST endpoints over GraphQL for internal services (knowledge/preference)
[End Knowledge Graph]
```

Keep it under 500-1000 tokens. Include:
- Node category (in parentheses)
- Content (plain text)
- Optional: subcategory

## Query Log Fields

| Field | Description |
|-------|-------------|
| `timestamp` | When the query was made (unix ms) |
| `query` | The search query text (for gaps analysis) |
| `query_type` | `search`, `kg_add`, `kg_link`, `kg_neighbors`, `kg_delete` |
| `results_returned` | Number of results (0 = gap in the graph) |
| `relevance_score` | The composite ranking score |
| `injected_ids` | Which nodes were injected into context (JSON array) |
| `injected_token_budget` | How many tokens were injected |
| `agent_action` | `used`, `ignored`, `acted-on` (tracked post-hoc) |

## "Done" Checklist

- [x] `normalize.ts` — `normalizeSubcategory("PR") === "pull-request"` ✓
- [x] `normalize.ts` — `normalizeSubcategory("pull_request") === "pull-request"` ✓
- [x] `normalize.ts` — `normalizeEdgeType("blocks") === normalizeEdgeType("blocked-by") === "blocks"` ✓
- [x] `kg_add` creates nodes with correct IDs, hashes, normalized subcategories
- [x] `kg_add` deduplicates by `content_hash` (updates instead of creating duplicate)
- [x] `kg_search` returns results ranked by BM25 + frequency + freshness
- [x] `kg_search` filters by `categories` and `subcategories` correctly
- [x] `kg_link` normalizes edge types and creates edges correctly
- [x] `kg_neighbors` traverses edges correctly (BFS/DFS)
- [x] `kg_delete` removes nodes and cascades to edges
- [x] `kg_get` returns full node + edge details
- [x] `kg_query` returns analytics from the query log
- [x] `/kg` command shows graph stats and category distribution
- [x] `/kg-query` command shows query log analytics
- [x] Session start injects top-N facts into system prompt
- [x] Compaction hook injects graph summary
- [x] Query log records every operation
- [x] nomic-embed-text-v1.5 integration (LMStudio fallback to FTS5)
- [x] Cosine distance normalized to [0, 1] via `/ 2.0` before combining with BM25
- [x] All parameters configurable via `kgMemory` settings key (graphPath, embeddingEndpoint, embeddingModel, maxResults, injectionBudget, queryLogLimit, staleNodeDays, ftxF5Weight, vectorWeight, frequencyWeight)
- [x] Falls back to FTS5 if LMStudio is down (default server at `10.1.1.145:1234`, configurable)
- [x] Database file is created on first run
- [x] All tests pass (216 automated tests in `test/`)
- [x] Test suite covers: normalization, CRUD, FTS5 search, vector search (stub), all 7 tools, all 5 hooks, logging/analytics, config validation

## Remaining / Future Enhancements

- [ ] Unit test suite (automated)
- [ ] Graph visualization in TUI (`/kg-graph`)
- [ ] Auto-discovery: scan session messages for facts and suggest graph additions
- [ ] Confidence scoring on nodes (based on source reliability)
- [ ] Superseded node tracking (old facts replaced by new ones)
- [ ] User preferences stored in graph (learning from corrections)
- [ ] Cross-project pattern detection (common patterns across projects)
- [ ] Graph diff between sessions (what changed?)
- [ ] Periodic maintenance: prune stale nodes, update embeddings, re-rank
- [ ] Allow users to add custom subcategory synonyms to `normalize.ts`
