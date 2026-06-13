/**
 * Knowledge Graph — Database Layer
 *
 * SQLite database with FTS5 and sqlite-vec support.
 * Handles schema creation, CRUD operations, and graph analytics.
 */

import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { createHash } from 'crypto';

// ---------------------------------------------------------------------------
// Schema (mirrors IMPLEMENTATION_PLAN.md)
// ---------------------------------------------------------------------------

const SCHEMA_SQL = `
-- Core nodes (category + subcategory + content)
CREATE TABLE IF NOT EXISTS nodes (
  id            TEXT PRIMARY KEY,
  category      TEXT NOT NULL,
  subcategory   TEXT,
  content       TEXT NOT NULL,
  properties    TEXT,
  created_at    REAL,
  last_accessed REAL,
  frequency     INTEGER DEFAULT 0,
  content_hash  TEXT
);

-- FTS5 virtual table for full-text search (BM25)
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
  USING fts5(content, category, subcategory, properties, content_hash, content='nodes');

-- Vector embeddings (populated when LMStudio embedding API is available)
CREATE TABLE IF NOT EXISTS node_vectors (
  node_id   TEXT PRIMARY KEY REFERENCES nodes(id),
  embedding BLOB
);

-- Relationships between nodes (edge types normalized)
CREATE TABLE IF NOT EXISTS edges (
  source_id TEXT REFERENCES nodes(id),
  target_id TEXT REFERENCES nodes(id),
  type      TEXT NOT NULL,
  created_at REAL,
  frequency INTEGER DEFAULT 0,
  PRIMARY KEY (source_id, target_id, type)
);

-- Query log (rolling window, oldest pruned at limit)
CREATE TABLE IF NOT EXISTS query_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp       REAL,
  query           TEXT,
  query_type      TEXT,
  results_returned INTEGER,
  relevance_score REAL,
  injected_ids    TEXT,
  injected_token_budget INTEGER,
  agent_action    TEXT
);

-- Graph snapshots for versioning/time-travel
CREATE TABLE IF NOT EXISTS graph_snapshots (
  version     INTEGER PRIMARY KEY,
  timestamp   TEXT,
  node_count  INTEGER,
  edge_count  INTEGER,
  summary     TEXT
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
`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface KnowledgeNode {
  id: string;
  category: string;
  subcategory: string | null;
  content: string;
  properties: Record<string, string> | null;
  createdAt: number;
  lastAccessedAt: number;
  frequency: number;
  contentHash: string;
}

export interface KnowledgeEdge {
  sourceId: string;
  targetId: string;
  type: string;
  createdAt: number;
  frequency: number;
}

export interface SearchHit {
  node: KnowledgeNode;
  bm25Score: number;
  vectorScore: number | null;
  frequencyBoost: number;
  freshnessScore: number;
  compositeScore: number;
  edges: KnowledgeEdge[];
}

// ---------------------------------------------------------------------------
// Database class
// ---------------------------------------------------------------------------

export class KnowledgeGraphDB {
  private db: Database.Database;
  private dbPath: string;
  private vecLoaded: boolean;

  constructor(dbPath: string) {
    this.dbPath = dbPath;
    this.vecLoaded = false;

    // Ensure directory exists
    const dir = path.dirname(dbPath);
    fs.mkdirSync(dir, { recursive: true });

    // Open (or create) the database
    this.db = new Database(dbPath, {
      wal: true,
      journalMode: 'wal',
      timeout: 5000,
      readonly: false,
    });

    // Configure SQLite pragmas
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('cache_size = -64000'); // 64MB

    // Note: sqlite-vec extension is not needed for the current vector pipeline.
    // search.ts performs all embedding operations (generation, storage, cosine distance)
    // in pure JavaScript. sqlite-vec would only be needed for SQL-based vector queries.
    this.vecLoaded = false;

    // Execute schema
    this.db.exec(SCHEMA_SQL);

    // Initialize FTS5 if nodes table exists
    try {
      const hasFts = this.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_fts'").get() as { name: string } | undefined;
      if (hasFts) {
        // Rebuild FTS index from existing nodes (only if there are nodes)
        const nodeCount = (this.db.prepare('SELECT COUNT(*) AS cnt FROM nodes').get() as { cnt: number }).cnt;
        if (nodeCount > 0) {
          this.db.prepare('INSERT OR IGNORE INTO nodes_fts SELECT * FROM nodes').run();
          this.db.prepare('fts5_rebuild(nodes_fts)').run();
        }
      }
    } catch {
      // FTS5 might not be compiled in — that's okay, search will fall back
      console.warn('[kg-memory] FTS5 rebuild skipped (may not be compiled in)');
    }
  }

  // -----------------------------------------------------------------------
  // Node CRUD
  // -----------------------------------------------------------------------

  /**
   * Get a node by its ID. Updates last_accessed and frequency.
   */
  getNode(id: string): KnowledgeNode | null {
    const row = this.db.prepare(`
      SELECT id, category, subcategory, content, properties,
             created_at, last_accessed, frequency, content_hash
      FROM nodes WHERE id = ?
    `).get(id) as Record<string, any> | undefined;

    if (!row) return null;

    return this._rowToNode(row);
  }

  /**
   * Get a node by its content hash (for dedup).
   */
  getNodeByContentHash(contentHash: string): KnowledgeNode[] {
    const rows = this.db.prepare(`
      SELECT id, category, subcategory, content, properties,
             created_at, last_accessed, frequency, content_hash
      FROM nodes WHERE content_hash = ?
    `).all(contentHash) as Record<string, any>[];

    return rows.map(r => this._rowToNode(r));
  }

  /**
   * Save (insert or update) a node.
   * If a node with the same content_hash exists, update it instead.
   */
  saveNode(node: Omit<KnowledgeNode, 'id'> & { id?: string }): { id: string; created: boolean } {
    const now = Date.now();
    const contentHash = createHash('sha256').update(node.content).digest('hex');

    const propertiesStr = node.properties ? JSON.stringify(node.properties) : null;

    // Check for dedup by content_hash
    const existing = this.db.prepare(`
      SELECT id, category, subcategory, frequency, last_accessed
      FROM nodes WHERE content_hash = ?
    `).get(contentHash) as Record<string, any> | undefined;

    // Generate deterministic ID from category + subcategory + content
    const key = `${node.category}:${node.subcategory || ''}:${node.content}`;
    const nodeId = `node_${createHash('sha256').update(key).digest('hex').slice(0, 16)}`;

    const insertValues = {
      id: nodeId,
      category: node.category,
      subcategory: node.subcategory,
      content: node.content,
      properties: propertiesStr,
      created_at: now,
      last_accessed: now,
      frequency: 0,
      content_hash: contentHash,
    };

    if (existing) {
      // Update existing node: extend its lifetime, boost frequency
      this.db.prepare(`
        UPDATE nodes
        SET last_accessed = ?, frequency = frequency + 1,
            subcategory = COALESCE(?, subcategory),
            category = COALESCE(NULLIF(?, ''), category)
        WHERE content_hash = ?
      `).run(now, node.subcategory, node.category, contentHash);

      return { id: existing.id, created: false };
    }

    // Insert new node
    this.db.prepare(`
      INSERT INTO nodes (id, category, subcategory, content, properties,
                         created_at, last_accessed, frequency, content_hash)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      insertValues.id,
      insertValues.category,
      insertValues.subcategory,
      insertValues.content,
      insertValues.properties,
      insertValues.created_at,
      insertValues.last_accessed,
      insertValues.frequency,
      insertValues.content_hash,
    );

    // Update FTS5 index (external content model — insert into fts table referencing nodes by rowid)
    try {
      const rowid = (this.db.prepare('SELECT rowid FROM nodes WHERE id = ?').get(nodeId) as { rowid: number } | undefined)?.rowid;
      if (rowid) {
        this.db.prepare(`
          INSERT OR REPLACE INTO nodes_fts (rowid, content, category, subcategory, properties, content_hash)
          VALUES (?, ?, ?, ?, ?, ?)
        `).run(rowid, node.content, node.category, node.subcategory, propertiesStr, contentHash);
      }
    } catch {
      // FTS5 might not be available — skip silently
    }

    return { id: nodeId, created: true };
  }

  /**
   * Delete a node and all incident edges.
   */
  deleteNode(id: string): boolean {
    const node = this.getNode(id);
    if (!node) return false;

    // Remove all incident edges first (before deleting the node to avoid FK violations)
    this.db.prepare('DELETE FROM edges WHERE source_id = ? OR target_id = ?').run(id, id);

    // Remove vector embedding
    this.db.prepare('DELETE FROM node_vectors WHERE node_id = ?').run(id);

    const result = this.db.prepare('DELETE FROM nodes WHERE id = ?').run(id);

    // Remove from FTS5 (external content model — delete by rowid)
    try {
      const rowid = (this.db.prepare('SELECT rowid FROM nodes WHERE id = ?').get(id) as { rowid: number } | undefined)?.rowid;
      if (rowid) {
        this.db.prepare(`DELETE FROM nodes_fts WHERE rowid = ?`).run(rowid);
      }
    } catch {
      // FTS5 might not be available
    }

    return result.changes > 0;
  }

  // -----------------------------------------------------------------------
  // Edge CRUD
  // -----------------------------------------------------------------------

  getEdge(sourceId: string, targetId: string, type: string): KnowledgeEdge | null {
    const row = this.db.prepare(`
      SELECT source_id, target_id, type, created_at, frequency
      FROM edges WHERE source_id = ? AND target_id = ? AND type = ?
    `).get(sourceId, targetId, type) as Record<string, any> | undefined;

    if (!row) return null;
    return this._rowToEdge(row);
  }

  saveEdge(edge: Omit<KnowledgeEdge, 'createdAt'> & { createdAt?: number }): { created: boolean; edge: KnowledgeEdge } {
    const now = edge.createdAt || Date.now();

    // Check if edge exists (with normalized type)
    const existing = this.db.prepare(`
      SELECT source_id, target_id, type, created_at, frequency
      FROM edges WHERE source_id = ? AND target_id = ? AND type = ?
    `).get(edge.sourceId, edge.targetId, edge.type) as Record<string, any> | undefined;

    if (existing) {
      this.db.prepare('UPDATE edges SET frequency = frequency + 1 WHERE source_id = ? AND target_id = ? AND type = ?').run(
        edge.sourceId, edge.targetId, edge.type
      );
      return { created: false, edge: this._rowToEdge(existing) };
    }

    this.db.prepare(`
      INSERT INTO edges (source_id, target_id, type, created_at, frequency)
      VALUES (?, ?, ?, ?, 0)
    `).run(edge.sourceId, edge.targetId, edge.type, now);

    const edgeResult = this.getEdge(edge.sourceId, edge.targetId, edge.type)!;
    return { created: true, edge: edgeResult };
  }

  deleteEdge(sourceId: string, targetId: string, type: string): boolean {
    const result = this.db.prepare(
      'DELETE FROM edges WHERE source_id = ? AND target_id = ? AND type = ?'
    ).run(sourceId, targetId, type);
    return result.changes > 0;
  }

  /**
   * Get all edges for a node (both source and target).
   */
  getNodeEdges(nodeId: string): KnowledgeEdge[] {
    const rows = this.db.prepare(`
      SELECT source_id, target_id, type, created_at, frequency
      FROM edges WHERE source_id = ? OR target_id = ?
    `).all(nodeId, nodeId) as Record<string, any>[];
    return rows.map(r => this._rowToEdge(r));
  }

  // -----------------------------------------------------------------------
  // Node retrieval helpers (for search layer)
  // -----------------------------------------------------------------------

  /**
   * Get all nodes, optionally filtered by category/subcategory.
   * Used by the search layer when FTS5 returns 0 results.
   */
  getAllNodes(categories?: string[], subcategories?: string[]): KnowledgeNode[] {
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

    const rows = this.db.prepare(`
      SELECT id, category, subcategory, content, properties,
             created_at, last_accessed, frequency, content_hash
      FROM nodes ${whereClause}
    `).all(...params) as Record<string, any>[];

    return rows.map(row => this._rowToNode(row));
  }

  /**
   * Get the maximum frequency across all nodes (for normalization).
   */
  getMaxFrequency(): number {
    const row = this.db.prepare(
      'SELECT COALESCE(MAX(frequency), 1) AS max_freq FROM nodes',
    ).get() as { max_freq: number };
    return row.max_freq;
  }

  // -----------------------------------------------------------------------
  // Search (FTS5 + vector)
  // -----------------------------------------------------------------------

  /**
   * FTS5 search with BM25 ranking (Phase 1 — no embeddings).
   */
  searchFTS5(query: string, maxResults: number = 10, categories?: string[], subcategories?: string[]): SearchHit[] {
    // Build WHERE clause for filters
    const whereClauses: string[] = [];
    const params: any[] = [];

    if (categories && categories.length > 0) {
      const placeholders = categories.map(() => '?').join(',');
      whereClauses.push(`n.category IN (${placeholders})`);
      params.push(...categories);
    }

    if (subcategories && subcategories.length > 0) {
      const placeholders = subcategories.map(() => '?').join(',');
      whereClauses.push(`n.subcategory IN (${placeholders})`);
      params.push(...subcategories);
    }

    // Handle empty/whitespace query: return all nodes (optionally filtered)
    const trimmedQuery = (query || '').trim();
    if (trimmedQuery === '') {
      const whereClause = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : '';
      const allNodes = this.db.prepare(`
        SELECT n.id, n.category, n.subcategory, n.content, n.properties,
               n.created_at, n.last_accessed, n.frequency, n.content_hash
        FROM nodes n
        ${whereClause}
        ORDER BY n.created_at DESC
        LIMIT ?
      `).all(...params, maxResults) as Record<string, any>[];

      const maxFreqRow = this.db.prepare(
        'SELECT COALESCE(MAX(frequency), 1) AS max_freq FROM nodes'
      ).get() as { max_freq: number };
      const maxFreq = maxFreqRow.max_freq;
      const now = Date.now();

      return allNodes.map(row => {
        const node = this._rowToNode(row);
        const frequencyBoost = maxFreq > 1
          ? Math.min(Math.log(node.frequency + 1) / Math.log(maxFreq + 1), 1.0)
          : 0;
        const hoursSinceAccessed = (now - node.lastAccessedAt) / (1000 * 60 * 60);
        const freshnessScore = 1.0 / (1.0 + hoursSinceAccessed / 720);

        return {
          node,
          bm25Score: 1.0,
          vectorScore: null,
          frequencyBoost,
          freshnessScore,
          compositeScore: (frequencyBoost * 0.3) + (freshnessScore * 0.3),
          edges: this.getNodeEdges(node.id),
        };
      }).sort((a, b) => b.compositeScore - a.compositeScore).slice(0, maxResults);
    }

    // FTS5 search: match against content, category, subcategory, properties
    // With the external content model, we join on rowid (nodes.rowid = nodes_fts.rowid)
    const filterClause = whereClauses.length > 0 ? `AND ${whereClauses.join(' AND ')}` : '';

    // FTS5 treats hyphens as grouping operators, so replace with underscores
    // This ensures "auth-service" is tokenized as "authservice" matching the index
    const escapedQuery = trimmedQuery.replace(/-/g, '_');
    let ftsQuery: Record<string, any>[];

    try {
      ftsQuery = this.db.prepare(`
        SELECT n.id, n.category, n.subcategory, n.content, n.properties,
               n.created_at, n.last_accessed, n.frequency, n.content_hash,
               nodes_fts.rank AS bm25_rank
        FROM nodes n
        JOIN nodes_fts ON n.rowid = nodes_fts.rowid
        WHERE nodes_fts MATCH ?
        ${filterClause}
        ORDER BY bm25_rank ASC
        LIMIT ?
      `).all(escapedQuery, ...params, maxResults) as Record<string, any>[];
    } catch (err) {
      // FTS5's MATCH operator does not support parameter binding (?) —
      // the ? is passed literally to FTS5's query parser, which interprets
      // characters like ?, *, +, (, ) as FTS5 operators, causing syntax
      // errors (e.g., "fts5: syntax error near '?'").
      //
      // Fall back to a simple LIKE query when FTS5 fails.
      // This sacrifices BM25 ranking but ensures search still works.
      console.warn('[kg-memory] FTS5 query failed, falling back to LIKE search:', (err as Error).message);

      // Escape single quotes for SQL safety and use LIKE with wildcards.
      // Each word gets its own LIKE clause for partial matching.
      const words = escapedQuery.split(/\s+/).filter(w => w.length > 0);
      const likeClauses = words.map(w => `n.content LIKE '%${w.replace(/'/g, "''")}%'`);
      const likeClause = likeClauses.join(' OR ');
      const whereClause = whereClauses.length > 0
        ? `WHERE ${whereClauses.join(' AND ')} AND ${likeClause}`
        : `WHERE ${likeClause}`;

      ftsQuery = this.db.prepare(`
        SELECT n.id, n.category, n.subcategory, n.content, n.properties,
               n.created_at, n.last_accessed, n.frequency, n.content_hash
        FROM nodes n
        ${whereClause}
        ORDER BY n.created_at DESC
        LIMIT ?
      `).all(...params, maxResults) as Record<string, any>[];
    }

    // Get max frequency for normalization
    const maxFreqRow = this.db.prepare(`
      SELECT COALESCE(MAX(frequency), 1) AS max_freq FROM nodes
    `).get() as { max_freq: number };
    const maxFreq = maxFreqRow.max_freq;

    const now = Date.now();

    return ftsQuery.map(row => {
      const node = this._rowToNode(row);
      const bm25Raw = row.bm25_rank || 0;

      // BM25 is lower = more relevant. Normalize to [0, 1] where 1 = most relevant.
      // We use a simple inverse: 1 / (1 + |rank|)
      const bm25Score = 1.0 / (1.0 + Math.abs(bm25Raw));

      // Frequency boost: log-normalized
      const frequencyBoost = maxFreq > 1
        ? Math.min(Math.log(node.frequency + 1) / Math.log(maxFreq + 1), 1.0)
        : 0;

      // Freshness: exponential decay with 30-day half-life
      const hoursSinceAccessed = (now - node.lastAccessedAt) / (1000 * 60 * 60);
      const freshnessScore = 1.0 / (1.0 + hoursSinceAccessed / 720); // 30 days = 720 hours

      // Composite score (Phase 1 weights)
      const compositeScore = (bm25Score * 0.4) + (frequencyBoost * 0.3) + (freshnessScore * 0.3);

      return {
        node,
        bm25Score,
        vectorScore: null,
        frequencyBoost,
        freshnessScore,
        compositeScore,
        edges: this.getNodeEdges(node.id),
      };
    }).sort((a, b) => b.compositeScore - a.compositeScore).slice(0, maxResults);
  }

  // -----------------------------------------------------------------------
  // Neighbors (graph traversal)
  // -----------------------------------------------------------------------

  /**
   * BFS traversal through edges up to maxDepth.
   */
  getNeighbors(nodeId: string, maxDepth: number = 2): { nodeId: string; edges: KnowledgeEdge[] }[] {
    const visited = new Set<string>();
    const result: { nodeId: string; edges: KnowledgeEdge[] }[] = [];
    const queue: { id: string; depth: number }[] = [{ id: nodeId, depth: 0 }];

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;

      if (visited.has(id)) continue;
      visited.add(id);

      if (id !== nodeId) {
        const edges = this.getNodeEdges(id);
        result.push({ nodeId: id, edges });
      }

      if (depth < maxDepth) {
        const edges = this.getNodeEdges(id);
        for (const edge of edges) {
          const neighborId = edge.sourceId === id ? edge.targetId : edge.sourceId;
          if (!visited.has(neighborId)) {
            queue.push({ id: neighborId, depth: depth + 1 });
          }
        }
      }
    }

    return result;
  }

  // -----------------------------------------------------------------------
  // Query log
  // -----------------------------------------------------------------------

  logQuery(entry: {
    query: string;
    queryType: string;
    resultsReturned: number;
    relevanceScore?: number;
    injectedIds?: string[];
    injectedTokenBudget?: number;
    agentAction?: string;
  }): void {
    this.db.prepare(`
      INSERT INTO query_log (timestamp, query, query_type, results_returned,
                             relevance_score, injected_ids, injected_token_budget, agent_action)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      Date.now(),
      entry.query,
      entry.queryType,
      entry.resultsReturned,
      entry.relevanceScore ?? null,
      entry.injectedIds ? JSON.stringify(entry.injectedIds) : null,
      entry.injectedTokenBudget ?? null,
      entry.agentAction ?? null,
    );
  }

  /**
   * Prune old query log entries, keeping the most recent N.
   */
  pruneQueryLog(limit: number = 1000): void {
    const count = this.db.prepare('SELECT COUNT(*) AS cnt FROM query_log').get() as { cnt: number };
    if (count.cnt > limit) {
      const toDelete = count.cnt - limit;
      this.db.prepare(`DELETE FROM query_log WHERE id IN (SELECT id FROM query_log ORDER BY id ASC LIMIT ?)`).run(toDelete);
    }
  }

  /**
   * Get query log entries (for analytics).
   */
  getQueryLogEntries(limit: number = 100, queryType?: string): Record<string, any>[] {
    if (queryType) {
      return this.db.prepare(`
        SELECT * FROM query_log WHERE query_type = ? ORDER BY timestamp DESC LIMIT ?
      `).all(queryType, limit) as Record<string, any>[];
    }
    return this.db.prepare(`
      SELECT * FROM query_log ORDER BY timestamp DESC LIMIT ?
    `).all(limit) as Record<string, any>[];
  }

  // -----------------------------------------------------------------------
  // Analytics
  // -----------------------------------------------------------------------

  /**
   * Get node count (lightweight — used for early-exit checks).
   */
  getNodeCount(): number {
    const row = this.db.prepare('SELECT COUNT(*) AS cnt FROM nodes').get() as { cnt: number };
    return row.cnt;
  }

  /**
   * Get graph statistics.
   */
  getGraphStats(): {
    nodeCount: number;
    edgeCount: number;
    oldestNode: string | null;
    newestNode: string | null;
    categoryDistribution: Record<string, number>;
    subcategoryDistribution: Record<string, number>;
    queryLogSize: number;
  } {
    const nodeCount = (this.db.prepare('SELECT COUNT(*) AS cnt FROM nodes').get() as { cnt: number }).cnt;
    const edgeCount = (this.db.prepare('SELECT COUNT(*) AS cnt FROM edges').get() as { cnt: number }).cnt;

    const oldestNode = (this.db.prepare(`
      SELECT id, content, category, subcategory
      FROM nodes ORDER BY created_at ASC LIMIT 1
    `).get() as { id: string; content: string; category: string; subcategory: string | null } | null);

    const newestNode = (this.db.prepare(`
      SELECT id, content, category, subcategory
      FROM nodes ORDER BY created_at DESC LIMIT 1
    `).get() as { id: string; content: string; category: string; subcategory: string | null } | null);

    const catRows = this.db.prepare(`
      SELECT category, COUNT(*) AS cnt FROM nodes GROUP BY category ORDER BY cnt DESC
    `).all() as { category: string; cnt: number }[];

    const subcatRows = this.db.prepare(`
      SELECT COALESCE(subcategory, '(none)') AS subcategory, COUNT(*) AS cnt
      FROM nodes GROUP BY subcategory ORDER BY cnt DESC
    `).all() as { subcategory: string; cnt: number }[];

    const queryLogSize = (this.db.prepare('SELECT COUNT(*) AS cnt FROM query_log').get() as { cnt: number }).cnt;

    return {
      nodeCount,
      edgeCount,
      oldestNode: oldestNode ? `${oldestNode.id} (${oldestNode.category}${oldestNode.subcategory ? `/${oldestNode.subcategory}` : ''})` : null,
      newestNode: newestNode ? `${newestNode.id} (${newestNode.category}${newestNode.subcategory ? `/${newestNode.subcategory}` : ''})` : null,
      categoryDistribution: Object.fromEntries(catRows.map(r => [r.category, r.cnt])),
      subcategoryDistribution: Object.fromEntries(subcatRows.map(r => [r.subcategory, r.cnt])),
      queryLogSize,
    };
  }

  /**
   * Most frequently surfaced nodes (by query log hits).
   */
  getMostSurfacedNodes(limit: number = 20): { nodeId: string; hits: number }[] {
    const rows = this.db.prepare(`
      SELECT injected_ids, results_returned
      FROM query_log WHERE query_type = 'search' AND injected_ids IS NOT NULL AND results_returned > 0
    `).all() as { injected_ids: string; results_returned: number }[];

    const nodeHits = new Map<string, number>();
    for (const row of rows) {
      try {
        const ids = JSON.parse(row.injected_ids) as string[];
        const perNodeHits = Math.floor(row.results_returned / Math.max(ids.length, 1));
        for (const id of ids) {
          nodeHits.set(id, (nodeHits.get(id) || 0) + perNodeHits);
        }
      } catch {
        // Malformed JSON — skip
      }
    }

    return Array.from(nodeHits.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([nodeId, hits]) => ({ nodeId, hits }));
  }

  /**
   * Zero-result queries (gaps in the graph).
   */
  getZeroResultQueries(limit: number = 20): { query: string; count: number }[] {
    return this.db.prepare(`
      SELECT query, COUNT(*) AS count
      FROM query_log WHERE query_type = 'search' AND results_returned = 0
      GROUP BY query ORDER BY count DESC LIMIT ?
    `).all(limit) as { query: string; count: number }[];
  }

  /**
   * Category distribution of nodes.
   */
  getCategoryDistribution(): { category: string; subcategory: string; count: number }[] {
    return this.db.prepare(`
      SELECT category, COALESCE(subcategory, '(none)') AS subcategory, COUNT(*) AS count
      FROM nodes GROUP BY category, subcategory ORDER BY count DESC
    `).all() as { category: string; subcategory: string; count: number }[];
  }

  // -----------------------------------------------------------------------
  // Snapshots
  // -----------------------------------------------------------------------

  createSnapshot(summary: string = ''): number {
    const stats = this.getGraphStats();
    const version = (this.db.prepare('SELECT COALESCE(MAX(version), 0) + 1 AS v FROM graph_snapshots').get() as { v: number }).v;

    this.db.prepare(`
      INSERT INTO graph_snapshots (version, timestamp, node_count, edge_count, summary)
      VALUES (?, ?, ?, ?, ?)
    `).run(
      version,
      new Date().toISOString(),
      stats.nodeCount,
      stats.edgeCount,
      summary,
    );

    return version;
  }

  getSnapshots(limit: number = 10): Record<string, any>[] {
    return this.db.prepare(`
      SELECT * FROM graph_snapshots ORDER BY version DESC LIMIT ?
    `).all(limit) as Record<string, any>[];
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  private _rowToNode(row: Record<string, any>): KnowledgeNode {
    return {
      id: row.id,
      category: row.category,
      subcategory: row.subcategory,
      content: row.content,
      properties: row.properties ? JSON.parse(row.properties) : null,
      createdAt: row.created_at,
      lastAccessedAt: row.last_accessed,
      frequency: row.frequency,
      contentHash: row.content_hash,
    };
  }

  private _rowToEdge(row: Record<string, any>): KnowledgeEdge {
    return {
      sourceId: row.source_id,
      targetId: row.target_id,
      type: row.type,
      createdAt: row.created_at,
      frequency: row.frequency,
    };
  }

  /**
   * Close the database connection.
   */
  close(): void {
    try {
      this.db.close();
    } catch {
      // Already closed
    }
  }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

export function openKnowledgeGraph(dbPath: string): KnowledgeGraphDB {
  return new KnowledgeGraphDB(dbPath);
}
