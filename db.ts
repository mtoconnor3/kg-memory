/**
 * Knowledge Graph — Database Layer
 *
 * SQLite database with FTS5 support.
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
  source        TEXT DEFAULT 'agent',
  trust         TEXT DEFAULT 'high',
  created_at    REAL,
  last_accessed REAL,
  access_count  INTEGER DEFAULT 0,
  frequency     INTEGER DEFAULT 0,
  content_hash  TEXT
);

-- FTS5 virtual table for full-text search (BM25)
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
  USING fts5(content, category, subcategory, properties, content_hash, content='nodes');

-- Vector embeddings (populated when LMStudio embedding API is available)
CREATE TABLE IF NOT EXISTS node_vectors (
  node_id   TEXT PRIMARY KEY REFERENCES nodes(id),
  embedding BLOB,
  model     TEXT,
  dim       INTEGER
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

-- Session markers (replaces graph_snapshots)
CREATE TABLE IF NOT EXISTS session_markers (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp   TEXT,
  event       TEXT,
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
  source: string;
  trust: string;
  createdAt: number;
  lastAccessedAt: number;
  accessCount: number;
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

  constructor(dbPath: string) {
    this.dbPath = dbPath;

    // Ensure directory exists
    const dir = path.dirname(dbPath);
    fs.mkdirSync(dir, { recursive: true });

    // Open (or create) the database — WAL set via pragma, not constructor
    this.db = new Database(dbPath, {
      timeout: 5000,
    });

    // Configure SQLite pragmas
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('cache_size = -64000'); // 64MB

    // Execute schema
    this.db.exec(SCHEMA_SQL);

    // Run migrations (add columns if they don't exist)
    this.migrate();

    // Initialize FTS5 if nodes table exists
    this.initFts();
  }

  // -----------------------------------------------------------------------
  // Migrations
  // -----------------------------------------------------------------------

  private migrate(): void {
    // Add 'source' and 'trust' columns if they don't exist (HK-3 provenance)
    try {
      this.db.pragma('table_info(nodes)');
      const columns = this.db.pragma('table_info(nodes)') as Array<{ name: string }>;
      const colNames = columns.map(c => c.name);

      if (!colNames.includes('source')) {
        this.db.exec('ALTER TABLE nodes ADD COLUMN source TEXT DEFAULT \'agent\'');
      }
      if (!colNames.includes('trust')) {
        this.db.exec('ALTER TABLE nodes ADD COLUMN trust TEXT DEFAULT \'high\'');
      }
      if (!colNames.includes('access_count')) {
        this.db.exec('ALTER TABLE nodes ADD COLUMN access_count INTEGER DEFAULT 0');
        this.db.exec('CREATE INDEX IF NOT EXISTS idx_nodes_access_count ON nodes(access_count)');
      }
    } catch {
      // Table might not exist yet — that's fine, schema will create it
    }

    // Add 'model' and 'dim' columns to node_vectors if they don't exist (SR-2)
    try {
      const vecColumns = this.db.pragma('table_info(node_vectors)') as Array<{ name: string }>;
      const vecColNames = vecColumns.map(c => c.name);

      if (!vecColNames.includes('model')) {
        this.db.exec('ALTER TABLE node_vectors ADD COLUMN model TEXT');
        // Backfill existing vectors with known model
        this.db.prepare('UPDATE node_vectors SET model = \'nomic-embed-text-v1.5\', dim = 768 WHERE model IS NULL').run();
      }
      if (!vecColNames.includes('dim')) {
        this.db.exec('ALTER TABLE node_vectors ADD COLUMN dim INTEGER');
        this.db.prepare('UPDATE node_vectors SET dim = 768 WHERE dim IS NULL').run();
      }
    } catch {
      // Table might not exist yet
    }

    // Rename graph_snapshots to session_markers if old table exists (DB-6)
    try {
      const hasSnapshots = this.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_snapshots'").get() as { name: string } | undefined;
      if (hasSnapshots) {
        // Migrate data to new table
        this.db.exec(`
          INSERT OR IGNORE INTO session_markers (timestamp, event, node_count, edge_count, summary)
          SELECT timestamp, 'snapshot', node_count, edge_count, summary FROM graph_snapshots
        `);
        this.db.exec('DROP TABLE graph_snapshots');
      }
    } catch {
      // Either table doesn't exist or migration already ran
    }
  }

  // -----------------------------------------------------------------------
  // FTS initialization
  // -----------------------------------------------------------------------

  private initFts(): void {
    try {
      const hasFts = this.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes_fts'").get() as { name: string } | undefined;
      if (hasFts) {
        // Rebuild FTS index from existing nodes using the correct external-content idiom
        const nodeCount = (this.db.prepare('SELECT COUNT(*) AS cnt FROM nodes').get() as { cnt: number }).cnt;
        if (nodeCount > 0) {
          this.db.prepare("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')").run();
        }
      }
    } catch {
      // FTS5 might not be compiled in — that's okay, search will fall back
      console.warn('[kg-memory] FTS5 rebuild skipped (may not be compiled in)');
    }
  }

  // -----------------------------------------------------------------------
  // Public DB accessor (for modules that need direct SQL — interim fix for TS-2)
  // -----------------------------------------------------------------------

  /**
   * Get the raw Database handle. Use sparingly — prefer typed methods above.
   * @deprecated Use typed methods instead; this exists for backward compatibility.
   */
  public getDb(): Database.Database {
    return this.db;
  }

  // -----------------------------------------------------------------------
  // Node CRUD
  // -----------------------------------------------------------------------

  /**
   * Get a node by its ID. Updates last_accessed and access_count.
   */
  getNode(id: string): KnowledgeNode | null {
    const row = this.db.prepare(`
      SELECT id, category, subcategory, content, properties,
             source, trust, created_at, last_accessed, access_count, frequency, content_hash
      FROM nodes WHERE id = ?
    `).get(id) as Record<string, any> | undefined;

    if (!row) return null;

    // Update access tracking (DB-3)
    const now = Date.now();
    this.db.prepare(`
      UPDATE nodes SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?
    `).run(now, id);

    return this._rowToNode(row);
  }

  /**
   * Save (insert or update) a node.
   * Identity is (category, subcategory, content) — no category mutation on dedup.
   * Only category, subcategory, content, and properties are required; the rest are auto-filled.
   */
  saveNode(node: {
    category: string;
    subcategory?: string | null;
    content: string;
    properties?: Record<string, string> | null;
    source?: string;
    trust?: string;
  }): { id: string; created: boolean } {
    const now = Date.now();
    const contentHash = createHash('sha256').update(node.content).digest('hex');

    const propertiesStr = node.properties ? JSON.stringify(node.properties) : null;

    // Generate deterministic ID from category + subcategory + content (DB-4)
    const key = `${node.category}:${node.subcategory || ''}:${node.content}`;
    const nodeId = `node_${createHash('sha256').update(key).digest('hex').slice(0, 16)}`;

    // Check for existing node with the same identity key (category + subcategory + content)
    const existing = this.db.prepare(`
      SELECT id, category, subcategory, frequency, last_accessed, access_count
      FROM nodes WHERE id = ?
    `).get(nodeId) as Record<string, any> | undefined;

    const insertValues = {
      id: nodeId,
      category: node.category,
      subcategory: node.subcategory,
      content: node.content,
      properties: propertiesStr,
      source: node.source || 'agent',
      trust: node.trust || 'high',
      created_at: now,
      last_accessed: now,
      access_count: 0,
      frequency: 0,
      content_hash: contentHash,
    };

    if (existing) {
      // Update existing node: bump frequency, update access, preserve category (DB-4)
      this.db.prepare(`
        UPDATE nodes
        SET last_accessed = ?, frequency = frequency + 1,
            access_count = access_count + 1,
            subcategory = COALESCE(NULLIF(?, ''), subcategory),
            properties = COALESCE(NULLIF(?, ''), properties)
        WHERE id = ?
      `).run(now, node.subcategory || null, propertiesStr || null, nodeId);

      return { id: existing.id, created: false };
    }

    // Insert new node
    this.db.prepare(`
      INSERT INTO nodes (id, category, subcategory, content, properties,
                         source, trust, created_at, last_accessed, access_count, frequency, content_hash)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      insertValues.id,
      insertValues.category,
      insertValues.subcategory,
      insertValues.content,
      insertValues.properties,
      insertValues.source,
      insertValues.trust,
      insertValues.created_at,
      insertValues.last_accessed,
      insertValues.access_count,
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

  /**
   * Get outgoing edges for a node (directional).
   */
  getOutgoingEdges(nodeId: string): KnowledgeEdge[] {
    const rows = this.db.prepare(`
      SELECT source_id, target_id, type, created_at, frequency
      FROM edges WHERE source_id = ?
    `).all(nodeId) as Record<string, any>[];
    return rows.map(r => this._rowToEdge(r));
  }

  /**
   * Get incoming edges for a node (directional).
   */
  getIncomingEdges(nodeId: string): KnowledgeEdge[] {
    const rows = this.db.prepare(`
      SELECT source_id, target_id, type, created_at, frequency
      FROM edges WHERE target_id = ?
    `).all(nodeId) as Record<string, any>[];
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
             source, trust, created_at, last_accessed, access_count, frequency, content_hash
      FROM nodes ${whereClause}
    `).all(...params) as Record<string, any>[];

    return rows.map(row => this._rowToNode(row));
  }

  /**
   * Get nodes for fallback embedding (capped, sorted by recency/frequency).
   * Used when FTS5 returns 0 results — prevents embedding all nodes (SR-1).
   */
  getFallbackCandidates(limit: number = 50, categories?: string[], subcategories?: string[]): KnowledgeNode[] {
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
             source, trust, created_at, last_accessed, access_count, frequency, content_hash
      FROM nodes ${whereClause}
      ORDER BY (frequency + access_count) DESC, last_accessed DESC
      LIMIT ?
    `).all(...params, limit) as Record<string, any>[];

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

  /**
   * Get the maximum access_count across all nodes (for normalization).
   */
  getMaxAccessCount(): number {
    const row = this.db.prepare(
      'SELECT COALESCE(MAX(access_count), 1) AS max_access FROM nodes',
    ).get() as { max_access: number };
    return row.max_access;
  }

  // -----------------------------------------------------------------------
  // Search (FTS5)
  // -----------------------------------------------------------------------

  /**
   * Sanitize a query string for FTS5 MATCH.
   * Wraps each term in double quotes to treat it as a literal phrase,
   * escaping any embedded quotes. This handles operator characters (-, *, (, ), etc.)
   * without losing BM25 ranking.
   */
  sanitizeFtsQuery(query: string): string {
    // Split on whitespace, wrap each token in quotes, escape embedded quotes
    const tokens = query.trim().split(/\s+/).filter(t => t.length > 0);
    return tokens
      .map(t => `"${t.replace(/"/g, '""')}"`)
      .join(' ');
  }

  /**
   * FTS5 search with BM25 ranking.
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
               n.source, n.trust, n.created_at, n.last_accessed, n.access_count, n.frequency, n.content_hash
        FROM nodes n
        ${whereClause}
        ORDER BY n.created_at DESC
        LIMIT ?
      `).all(...params, maxResults) as Record<string, any>[];

      return this.scoreHits(allNodes, 0, null, maxResults);
    }

    // FTS5 search: match against content, category, subcategory, properties
    const filterClause = whereClauses.length > 0 ? `AND ${whereClauses.join(' AND ')}` : '';

    // Sanitize query for FTS5 MATCH — handles operator chars without losing BM25 (DB-7, DB-8)
    const sanitizedQuery = this.sanitizeFtsQuery(trimmedQuery);

    const ftsResults = this.db.prepare(`
      SELECT n.id, n.category, n.subcategory, n.content, n.properties,
             n.source, n.trust, n.created_at, n.last_accessed, n.access_count, n.frequency, n.content_hash,
             nodes_fts.rank AS bm25_rank
      FROM nodes n
      JOIN nodes_fts ON n.rowid = nodes_fts.rowid
      WHERE nodes_fts MATCH ?
      ${filterClause}
      ORDER BY bm25_rank ASC
      LIMIT ?
    `).all(sanitizedQuery, ...params, maxResults) as Record<string, any>[];

    return this.scoreHits(ftsResults, 1.0 / (1.0 + Math.abs(ftsResults.length > 0 ? (ftsResults[0] as any).bm25_rank : 0)), null, maxResults);
  }

  /**
   * Score a set of node rows into SearchHits.
   * Single unified scoring function (SR-3).
   */
  private scoreHits(
    rows: Record<string, any>[],
    _bm25Ref: number,
    _vectorRef: number | null,
    maxResults: number,
  ): SearchHit[] {
    const maxFreq = this.getMaxFrequency();
    const maxAccess = this.getMaxAccessCount();
    const now = Date.now();

    return rows.map(row => {
      const node = this._rowToNode(row);

      // BM25 score (from FTS rank, or 0 for non-FTS paths)
      const bm25Raw = row.bm25_rank || 0;
      const bm25Score = 1.0 / (1.0 + Math.abs(bm25Raw));

      // Frequency boost: log-normalized using access_count (DB-5)
      const accessScore = maxAccess > 1
        ? Math.min(Math.log(node.accessCount + 1) / Math.log(maxAccess + 1), 1.0)
        : 0;
      const frequencyBoost = maxFreq > 1
        ? Math.min(Math.log(node.frequency + 1) / Math.log(maxFreq + 1), 1.0)
        : 0;

      // Freshness: exponential decay with 30-day half-life
      const hoursSinceAccessed = (now - node.lastAccessedAt) / (1000 * 60 * 60);
      const freshnessScore = 1.0 / (1.0 + hoursSinceAccessed / 720);

      return {
        node,
        bm25Score,
        vectorScore: null,
        frequencyBoost,
        freshnessScore,
        compositeScore: 0, // Set by caller with proper weights
        edges: this.getNodeEdges(node.id),
      };
    }).sort((a, b) => b.compositeScore - a.compositeScore).slice(0, maxResults);
  }

  // -----------------------------------------------------------------------
  // Neighbors (graph traversal)
  // -----------------------------------------------------------------------

  /**
   * BFS traversal through edges up to maxDepth.
   * Fetches edges once per node (DB-9).
   */
  getNeighbors(nodeId: string, maxDepth: number = 2): { nodeId: string; edges: KnowledgeEdge[] }[] {
    const visited = new Set<string>();
    const result: { nodeId: string; edges: KnowledgeEdge[] }[] = [];
    const queue: { id: string; depth: number }[] = [{ id: nodeId, depth: 0 }];

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;

      if (visited.has(id)) continue;
      visited.add(id);

      // Fetch edges once per node (DB-9)
      const edges = this.getNodeEdges(id);

      if (id !== nodeId) {
        result.push({ nodeId: id, edges });
      }

      if (depth < maxDepth) {
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
  pruneQueryLog(limit: number = 1000): number {
    const count = this.db.prepare('SELECT COUNT(*) AS cnt FROM query_log').get() as { cnt: number };
    if (count.cnt > limit) {
      const toDelete = count.cnt - limit;
      this.db.prepare(`DELETE FROM query_log WHERE id IN (SELECT id FROM query_log ORDER BY id ASC LIMIT ?)`).run(toDelete);
      return toDelete;
    }
    return 0;
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
  // Analytics (single source of truth — LG-1)
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

  /**
   * Query type distribution (LG-2: aligned field names).
   */
  getQueryTypeDistribution(): { queryType: string; count: number }[] {
    return this.db.prepare(`
      SELECT query_type AS queryType, COUNT(*) AS count
      FROM query_log
      GROUP BY query_type
      ORDER BY count DESC
    `).all() as { queryType: string; count: number }[];
  }

  /**
   * Agent action distribution (LG-2: aligned field names).
   */
  getAgentActionDistribution(): { action: string; count: number }[] {
    return this.db.prepare(`
      SELECT agent_action AS action, COUNT(*) AS count
      FROM query_log
      WHERE agent_action IS NOT NULL
      GROUP BY agent_action
      ORDER BY count DESC
    `).all() as { action: string; count: number }[];
  }

  /**
   * Graph growth over time (LG-3: true cumulative).
   */
  getGraphGrowth(limit: number = 30): { date: string; nodeCount: number }[] {
    const rows = this.db.prepare(`
      SELECT DATE(created_at / 1000, 'unixepoch') AS date, COUNT(*) AS count
      FROM nodes
      GROUP BY date
      ORDER BY date ASC
    `).all() as Array<{ date: string; count: number }>;

    // True cumulative total (LG-3)
    let cumulative = 0;
    return rows.map(row => {
      cumulative += row.count;
      return { date: row.date, nodeCount: cumulative };
    }).slice(-limit);
  }

  // -----------------------------------------------------------------------
  // Session markers (replaces graph_snapshots — DB-6)
  // -----------------------------------------------------------------------

  createMarker(event: string, summary: string = ''): number {
    const stats = this.getGraphStats();

    this.db.prepare(`
      INSERT INTO session_markers (timestamp, event, node_count, edge_count, summary)
      VALUES (?, ?, ?, ?, ?)
    `).run(
      new Date().toISOString(),
      event,
      stats.nodeCount,
      stats.edgeCount,
      summary,
    );

    const row = this.db.prepare('SELECT last_insert_rowid() AS id').get() as { id: number };
    return row.id;
  }

  getMarkers(limit: number = 10): Record<string, any>[] {
    return this.db.prepare(`
      SELECT * FROM session_markers ORDER BY id DESC LIMIT ?
    `).all(limit) as Record<string, any>[];
  }

  pruneMarkers(keepLast: number = 50): void {
    // Find the (keepLast+1)th oldest marker and delete everything up to (but not including) it
    const row = this.db.prepare(`
      SELECT id FROM session_markers ORDER BY id ASC LIMIT 1 OFFSET ?
    `).get(keepLast) as { id: number } | undefined;
    if (row) {
      this.db.prepare('DELETE FROM session_markers WHERE id < ?').run(row.id);
    }
  }

  // -----------------------------------------------------------------------
  // Vector storage (for search layer — SR-2)
  // -----------------------------------------------------------------------

  storeVector(nodeId: string, embedding: number[], model: string): void {
    const buffer = Buffer.from(new Float32Array(embedding).buffer);
    try {
      this.db.prepare(
        'INSERT OR REPLACE INTO node_vectors (node_id, embedding, model, dim) VALUES (?, ?, ?, ?)'
      ).run(nodeId, buffer, model, embedding.length);
    } catch (err) {
      console.warn('[kg-memory] Failed to store embedding:', (err as Error).message);
    }
  }

  getVector(nodeId: string): { embedding: number[]; model: string; dim: number } | null {
    try {
      const row = this.db.prepare(
        'SELECT embedding, model, dim FROM node_vectors WHERE node_id = ?'
      ).get(nodeId) as { embedding: Buffer; model: string; dim: number } | undefined;

      if (!row) return null;

      const floatArray = new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4);
      return {
        embedding: Array.from(floatArray),
        model: row.model,
        dim: row.dim,
      };
    } catch {
      return null;
    }
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
      source: row.source || 'agent',
      trust: row.trust || 'high',
      createdAt: row.created_at,
      lastAccessedAt: row.last_accessed,
      accessCount: row.access_count || 0,
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
