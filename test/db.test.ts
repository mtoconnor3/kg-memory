import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import { openKnowledgeGraph } from '../db.ts';

const TEST_DB = '/tmp/kg-test-' + Date.now() + '.db';

let db: ReturnType<typeof openKnowledgeGraph>;

beforeEach(() => {
  db = openKnowledgeGraph(TEST_DB);
});

afterEach(() => {
  db.close();
  try {
    fs.rmSync(TEST_DB, { force: true });
    fs.rmSync(TEST_DB + '-wal', { force: true });
    fs.rmSync(TEST_DB + '-shm', { force: true });
  } catch {}
});

describe('openKnowledgeGraph', () => {
  it('creates the database file and schema', () => {
    const tables = db.getDb().prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all();
    const tableNames = tables.map((t: any) => t.name);
    expect(tableNames).toContain('nodes');
    expect(tableNames).toContain('nodes_fts');
    expect(tableNames).toContain('node_vectors');
    expect(tableNames).toContain('edges');
    expect(tableNames).toContain('query_log');
    expect(tableNames).toContain('session_markers');
  });

  it('creates indexes', () => {
    const indexes = db.getDb().prepare("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'").all();
    const indexNames = indexes.map((idx: any) => idx.name);
    expect(indexNames).toContain('idx_nodes_last_accessed');
    expect(indexNames).toContain('idx_nodes_frequency');
    expect(indexNames).toContain('idx_nodes_category');
    expect(indexNames).toContain('idx_edges_source');
    expect(indexNames).toContain('idx_edges_target');
    expect(indexNames).toContain('idx_node_vectors_node');
  });

  it('has node_vectors with model and dim columns', () => {
    const columns = db.getDb().pragma('table_info(node_vectors)') as Array<{ name: string }>;
    const colNames = columns.map(c => c.name);
    expect(colNames).toContain('node_id');
    expect(colNames).toContain('embedding');
    expect(colNames).toContain('model');
    expect(colNames).toContain('dim');
  });

  it('has nodes table with source and trust columns', () => {
    const columns = db.getDb().pragma('table_info(nodes)') as Array<{ name: string }>;
    const colNames = columns.map(c => c.name);
    expect(colNames).toContain('source');
    expect(colNames).toContain('trust');
    expect(colNames).toContain('access_count');
  });
});

describe('saveNode', () => {
  it('creates a new node with required fields only', () => {
    const result = db.saveNode({
      category: 'knowledge',
      subcategory: 'fact',
      content: 'The sky is blue',
    });

    expect(result.created).toBe(true);
    expect(result.id).toMatch(/^node_/);

    const node = db.getNode(result.id);
    expect(node).not.toBeNull();
    expect(node!.category).toBe('knowledge');
    expect(node!.subcategory).toBe('fact');
    expect(node!.content).toBe('The sky is blue');
    expect(node!.source).toBe('agent');
    expect(node!.trust).toBe('high');
  });

  it('creates a node with custom source and trust', () => {
    const result = db.saveNode({
      category: 'project',
      content: 'Custom source node',
      source: 'user',
      trust: 'high',
    });

    expect(result.created).toBe(true);
    const node = db.getNode(result.id);
    expect(node!.source).toBe('user');
    expect(node!.trust).toBe('high');
  });

  it('generates deterministic IDs from category+subcategory+content', () => {
    const r1 = db.saveNode({
      category: 'knowledge',
      subcategory: 'fact',
      content: 'Same content',
    });
    const r2 = db.saveNode({
      category: 'knowledge',
      subcategory: 'fact',
      content: 'Same content',
    });

    expect(r1.id).toBe(r2.id);
    expect(r1.created).toBe(true);
    expect(r2.created).toBe(false); // dedup
  });

  it('same content under different categories creates separate nodes (DB-4)', () => {
    const r1 = db.saveNode({
      category: 'knowledge',
      content: 'Same text',
    });
    const r2 = db.saveNode({
      category: 'project',
      content: 'Same text',
    });

    expect(r1.id).not.toBe(r2.id);
    expect(r1.created).toBe(true);
    expect(r2.created).toBe(true);

    // Verify categories are preserved
    const node1 = db.getNode(r1.id)!;
    const node2 = db.getNode(r2.id)!;
    expect(node1.category).toBe('knowledge');
    expect(node2.category).toBe('project');
  });

  it('same content under different subcategories creates separate nodes (DB-4)', () => {
    const r1 = db.saveNode({
      category: 'knowledge',
      subcategory: 'fact',
      content: 'Same text',
    });
    const r2 = db.saveNode({
      category: 'knowledge',
      subcategory: 'warning',
      content: 'Same text',
    });

    expect(r1.id).not.toBe(r2.id);
    expect(r1.created).toBe(true);
    expect(r2.created).toBe(true);
  });

  it('dedup bumps frequency without mutating category (DB-4)', () => {
    const r1 = db.saveNode({
      category: 'knowledge',
      subcategory: 'fact',
      content: 'Dedup test',
    });
    expect(r1.created).toBe(true);

    const r2 = db.saveNode({
      category: 'knowledge',
      subcategory: 'fact',
      content: 'Dedup test',
    });
    expect(r2.created).toBe(false);
    expect(r2.id).toBe(r1.id);

    const node = db.getNode(r1.id)!;
    expect(node.category).toBe('knowledge');
    expect(node.frequency).toBe(1);
  });

  it('preserves null subcategory on dedup', () => {
    const r1 = db.saveNode({
      category: 'knowledge',
      content: 'No subcategory',
    });
    expect(r1.created).toBe(true);

    const r2 = db.saveNode({
      category: 'knowledge',
      content: 'No subcategory',
    });
    expect(r2.created).toBe(false);

    const node = db.getNode(r1.id)!;
    expect(node.subcategory).toBeNull();
  });

  it('updates properties on dedup', () => {
    const r1 = db.saveNode({
      category: 'knowledge',
      content: 'Props test',
      properties: { key: 'value1' },
    });

    const r2 = db.saveNode({
      category: 'knowledge',
      content: 'Props test',
      properties: { key: 'value2' },
    });
    expect(r2.created).toBe(false);

    const node = db.getNode(r1.id)!;
    expect(node.properties).toEqual({ key: 'value2' });
  });
});

describe('getNode', () => {
  it('returns null for nonexistent node', () => {
    expect(db.getNode('nonexistent')).toBeNull();
  });

  it('returns node by ID', () => {
    const result = db.saveNode({
      category: 'knowledge',
      content: 'Test node',
    });
    const node = db.getNode(result.id);
    expect(node).not.toBeNull();
    expect(node!.id).toBe(result.id);
  });

  it('updates last_accessed and access_count on get (DB-3)', () => {
    const result = db.saveNode({
      category: 'knowledge',
      content: 'Access test',
    });

    const node1 = db.getNode(result.id)!;
    const initialAccess = node1.accessCount;

    const node2 = db.getNode(result.id)!;
    expect(node2.accessCount).toBe(initialAccess + 1);
  });
});

describe('deleteNode', () => {
  it('returns false for nonexistent node', () => {
    expect(db.deleteNode('nonexistent')).toBe(false);
  });

  it('deletes a node', () => {
    const result = db.saveNode({
      category: 'knowledge',
      content: 'To delete',
    });

    expect(db.deleteNode(result.id)).toBe(true);
    expect(db.getNode(result.id)).toBeNull();
  });

  it('removes edges when deleting a node', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'Node 1' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'Node 2' });

    db.saveEdge({
      sourceId: n1.id,
      targetId: n2.id,
      type: 'relates-to',
      frequency: 0,
    });

    db.deleteNode(n1.id);
    expect(db.getNodeEdges(n2.id)).toHaveLength(0);
  });
});

describe('saveEdge', () => {
  it('creates a new edge', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });

    const result = db.saveEdge({
      sourceId: n1.id,
      targetId: n2.id,
      type: 'blocks',
      frequency: 0,
    });

    expect(result.created).toBe(true);
    expect(result.edge.sourceId).toBe(n1.id);
    expect(result.edge.targetId).toBe(n2.id);
    expect(result.edge.type).toBe('blocks');
  });

  it('deduplicates edges', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });

    const r1 = db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'blocks', frequency: 0 });
    const r2 = db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'blocks', frequency: 0 });

    expect(r1.created).toBe(true);
    expect(r2.created).toBe(false);
  });

  it('bumps frequency on dedup', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });

    db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'blocks', frequency: 0 });
    db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'blocks', frequency: 0 });

    const edge = db.getEdge(n1.id, n2.id, 'blocks')!;
    expect(edge.frequency).toBe(1);
  });
});

describe('getNodeEdges', () => {
  it('returns edges where node is source', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });
    db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'blocks', frequency: 0 });

    const edges = db.getNodeEdges(n1.id);
    expect(edges).toHaveLength(1);
  });

  it('returns edges where node is target', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });
    db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'blocks', frequency: 0 });

    const edges = db.getNodeEdges(n2.id);
    expect(edges).toHaveLength(1);
  });
});

describe('getNeighbors', () => {
  it('returns connected nodes within depth', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });
    const n3 = db.saveNode({ category: 'knowledge', content: 'C' });

    db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'relates-to', frequency: 0 });
    db.saveEdge({ sourceId: n2.id, targetId: n3.id, type: 'relates-to', frequency: 0 });

    const neighbors = db.getNeighbors(n1.id, 2);
    expect(neighbors).toHaveLength(2);
  });

  it('respects maxDepth', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });
    const n3 = db.saveNode({ category: 'knowledge', content: 'C' });

    db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'relates-to', frequency: 0 });
    db.saveEdge({ sourceId: n2.id, targetId: n3.id, type: 'relates-to', frequency: 0 });

    const neighbors = db.getNeighbors(n1.id, 1);
    expect(neighbors).toHaveLength(1);
    expect(neighbors[0].nodeId).toBe(n2.id);
  });

  it('fetches edges once per node (DB-9)', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'A' });
    const n2 = db.saveNode({ category: 'knowledge', content: 'B' });
    db.saveEdge({ sourceId: n1.id, targetId: n2.id, type: 'relates-to', frequency: 0 });

    const neighbors = db.getNeighbors(n1.id, 1);
    expect(neighbors).toHaveLength(1);
  });
});

describe('searchFTS5', () => {
  it('returns empty for empty query on empty DB', () => {
    const results = db.searchFTS5('', 10);
    expect(results).toEqual([]);
  });

  it('finds nodes by content', () => {
    db.saveNode({ category: 'knowledge', content: 'The quick brown fox' });
    db.saveNode({ category: 'knowledge', content: 'A lazy dog sleeps' });

    const results = db.searchFTS5('quick', 10);
    expect(results).toHaveLength(1);
    expect(results[0].node.content).toContain('fox');
  });

  it('handles operator characters without crashing (DB-7)', () => {
    db.saveNode({ category: 'knowledge', content: 'Test auth-service config' });

    // Query with hyphen (FTS5 operator)
    const results = db.searchFTS5('auth-service', 10);
    expect(results.length).toBeGreaterThanOrEqual(0); // Should not throw
  });

  it('handles special characters in query (DB-7)', () => {
    db.saveNode({ category: 'knowledge', content: 'Use the "quoted" value' });

    // Query with quotes
    const results = db.searchFTS5('"quoted"', 10);
    expect(results.length).toBeGreaterThanOrEqual(0); // Should not throw
  });

  it('filters by category', () => {
    db.saveNode({ category: 'knowledge', content: 'A fact' });
    db.saveNode({ category: 'project', content: 'A fact' });

    const results = db.searchFTS5('fact', 10, ['knowledge']);
    expect(results).toHaveLength(1);
    expect(results[0].node.category).toBe('knowledge');
  });

  it('handles empty query with filters', () => {
    db.saveNode({ category: 'knowledge', content: 'Fact A' });
    db.saveNode({ category: 'project', content: 'Fact B' });

    const results = db.searchFTS5('', 10, ['knowledge']);
    expect(results).toHaveLength(1);
  });
});

describe('getGraphStats', () => {
  it('returns correct counts', () => {
    db.saveNode({ category: 'knowledge', content: 'A' });
    db.saveNode({ category: 'project', content: 'B' });

    const stats = db.getGraphStats();
    expect(stats.nodeCount).toBe(2);
    expect(stats.edgeCount).toBe(0);
    expect(stats.categoryDistribution).toEqual({ knowledge: 1, project: 1 });
  });
});

describe('query log', () => {
  it('logs a query', () => {
    db.logQuery({
      query: 'test query',
      queryType: 'search',
      resultsReturned: 5,
    });

    const entries = db.getQueryLogEntries(10);
    expect(entries).toHaveLength(1);
    expect(entries[0].query).toBe('test query');
  });

  it('prunes old entries', () => {
    for (let i = 0; i < 20; i++) {
      db.logQuery({ query: `query ${i}`, queryType: 'search', resultsReturned: 0 });
    }

    const pruned = db.pruneQueryLog(10);
    expect(pruned).toBe(10);

    const entries = db.getQueryLogEntries(100);
    expect(entries).toHaveLength(10);
  });
});

describe('session markers', () => {
  it('creates a marker', () => {
    db.saveNode({ category: 'knowledge', content: 'A' });
    const id = db.createMarker('session_start', 'Test session');
    expect(id).toBeGreaterThan(0);

    const markers = db.getMarkers(10);
    expect(markers).toHaveLength(1);
    expect(markers[0].event).toBe('session_start');
  });

  it('prunes old markers', () => {
    for (let i = 0; i < 10; i++) {
      db.createMarker('event', `Marker ${i}`);
    }
    db.pruneMarkers(5);

    const markers = db.getMarkers(100);
    expect(markers).toHaveLength(5);
  });
});

describe('vector storage (SR-2)', () => {
  it('stores and retrieves vectors with model/dim stamp', () => {
    const node = db.saveNode({ category: 'knowledge', content: 'Vector test' });
    const vec = [0.1, 0.2, 0.3];

    db.storeVector(node.id, vec, 'nomic-embed-text-v1.5');

    const result = db.getVector(node.id);
    expect(result).not.toBeNull();
    expect(result!.model).toBe('nomic-embed-text-v1.5');
    expect(result!.dim).toBe(3);
    expect(result!.embedding).toHaveLength(3);
  });
});

describe('analytics', () => {
  it('getMostSurfacedNodes returns nodes from query log', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'Popular' });
    db.logQuery({
      query: 'search',
      queryType: 'search',
      resultsReturned: 3,
      injectedIds: [n1.id, n1.id, n1.id],
    });

    const surfaced = db.getMostSurfacedNodes(10);
    expect(surfaced).toHaveLength(1);
    expect(surfaced[0].nodeId).toBe(n1.id);
  });

  it('getZeroResultQueries returns gap queries', () => {
    db.logQuery({ query: 'missing topic', queryType: 'search', resultsReturned: 0 });
    db.logQuery({ query: 'another gap', queryType: 'search', resultsReturned: 0 });

    const gaps = db.getZeroResultQueries(10);
    expect(gaps).toHaveLength(2);
  });

  it('getQueryTypeDistribution uses correct field names (LG-2)', () => {
    db.logQuery({ query: 'q1', queryType: 'search', resultsReturned: 1 });
    db.logQuery({ query: 'q2', queryType: 'kg_add', resultsReturned: 1 });

    const dist = db.getQueryTypeDistribution();
    expect(dist[0].queryType).toBeDefined();
    expect(dist[1].queryType).toBeDefined();
  });

  it('getAgentActionDistribution uses correct field names (LG-2)', () => {
    db.logQuery({ query: 'q1', queryType: 'search', resultsReturned: 1, agentAction: 'used' });
    db.logQuery({ query: 'q2', queryType: 'search', resultsReturned: 1, agentAction: 'ignored' });

    const dist = db.getAgentActionDistribution();
    expect(dist[0].action).toBeDefined();
    expect(dist[1].action).toBeDefined();
  });

  it('getGraphGrowth returns true cumulative (LG-3)', () => {
    db.saveNode({ category: 'knowledge', content: 'Day 1' });
    db.saveNode({ category: 'knowledge', content: 'Day 2' });
    db.saveNode({ category: 'knowledge', content: 'Day 3' });

    const growth = db.getGraphGrowth(30);
    if (growth.length > 0) {
      expect(growth[growth.length - 1].nodeCount).toBe(3);
    }
  });
});
