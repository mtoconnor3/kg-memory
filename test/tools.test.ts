import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import { openKnowledgeGraph } from '../db.ts';
import {
  kgAdd,
  kgSearch,
  kgLink,
  kgNeighbors,
  kgGet,
  kgDelete,
  kgQuery,
} from '../tools.ts';
import { DEFAULT_CONFIG, validateConfig } from '../search.ts';

const TEST_DB = '/tmp/kg-tools-test-' + Date.now() + '.db';

let db: ReturnType<typeof openKnowledgeGraph>;
const config = validateConfig({ embeddingEndpoint: 'http://localhost:65535/v1/embeddings' });

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

describe('kgAdd', () => {
  it('creates a new node', () => {
    const result = kgAdd(db, {
      category: 'knowledge',
      content: 'The sky is blue',
      subcategory: 'fact',
    });

    expect(result.success).toBe(true);
    expect(result.created).toBe(true);
    expect(result.nodeId).toMatch(/^node_/);
  });

  it('validates category', () => {
    const result = kgAdd(db, {
      category: 'invalid',
      content: 'Test',
    });

    expect(result.success).toBe(false);
    expect(result.message).toContain('Invalid category');
  });

  it('normalizes subcategory', () => {
    const result = kgAdd(db, {
      category: 'knowledge',
      content: 'A fact',
      subcategory: 'Facts',
    });

    expect(result.success).toBe(true);
    const node = db.getNode(result.nodeId!);
    expect(node!.subcategory).toBe('fact');
  });

  it('caps content at MAX_CONTENT_LENGTH (TL-4)', () => {
    const longContent = 'x'.repeat(5000);
    const result = kgAdd(db, {
      category: 'knowledge',
      content: longContent,
    });

    expect(result.success).toBe(true);
    const node = db.getNode(result.nodeId!);
    expect(node!.content.length).toBeLessThanOrEqual(4000 + 15); // truncated + suffix
    expect(node!.content).toContain('[truncated]');
  });

  it('deduplicates by identity', () => {
    const r1 = kgAdd(db, {
      category: 'knowledge',
      content: 'Same content',
      subcategory: 'fact',
    });
    const r2 = kgAdd(db, {
      category: 'knowledge',
      content: 'Same content',
      subcategory: 'fact',
    });

    expect(r1.created).toBe(true);
    expect(r2.created).toBe(false);
    expect(r1.nodeId).toBe(r2.nodeId);
  });
});

describe('kgSearch', () => {
  it('searches the graph', async () => {
    kgAdd(db, { category: 'knowledge', content: 'Hello world test' });

    const result = await kgSearch(db, { query: 'hello' }, config);
    expect(result.success).toBe(true);
    expect(result.results.length).toBeGreaterThanOrEqual(0);
  });

  it('handles search failure gracefully (TL-2)', async () => {
    // This should not throw even if searchHybrid fails
    const result = await kgSearch(db, { query: 'test' }, config);
    expect(result.success).toBeDefined();
  });

  it('logs the search operation', async () => {
    kgAdd(db, { category: 'knowledge', content: 'Test' });

    await kgSearch(db, { query: 'test' }, config);
    const entries = db.getQueryLogEntries(10);
    const searchEntry = entries.find(e => e.query_type === 'search');
    expect(searchEntry).toBeDefined();
  });
});

describe('kgLink', () => {
  it('creates an edge', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });

    const result = kgLink(db, {
      sourceId: n1.nodeId!,
      targetId: n2.nodeId!,
      type: 'blocks',
    });

    expect(result.success).toBe(true);
    expect(result.created).toBe(true);
  });

  it('validates source node exists', () => {
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });

    const result = kgLink(db, {
      sourceId: 'nonexistent',
      targetId: n2.nodeId!,
      type: 'blocks',
    });

    expect(result.success).toBe(false);
    expect(result.message).toContain('not found');
  });

  it('validates target node exists', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });

    const result = kgLink(db, {
      sourceId: n1.nodeId!,
      targetId: 'nonexistent',
      type: 'blocks',
    });

    expect(result.success).toBe(false);
    expect(result.message).toContain('not found');
  });

  it('normalizes edge type', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });

    const result = kgLink(db, {
      sourceId: n1.nodeId!,
      targetId: n2.nodeId!,
      type: 'DEP-ON',
    });

    expect(result.message).toContain('depends-on');
  });

  it('swaps endpoints for inverse types (NM-1)', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });

    // A blocked-by B → B blocks A
    const result = kgLink(db, {
      sourceId: n1.nodeId!,
      targetId: n2.nodeId!,
      type: 'blocked-by',
    });

    expect(result.message).toContain('blocks');
    // The edge should be stored as B → A (blocks)
    const edge = db.getEdge(n2.nodeId!, n1.nodeId!, 'blocks');
    expect(edge).not.toBeNull();
  });

  it('swaps for "uses" (A uses B → B used-by A)', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });

    const result = kgLink(db, {
      sourceId: n1.nodeId!,
      targetId: n2.nodeId!,
      type: 'uses',
    });

    // Should be stored as B used-by A
    const edge = db.getEdge(n2.nodeId!, n1.nodeId!, 'used-by');
    expect(edge).not.toBeNull();
  });

  it('swaps for "caused-by" (A caused-by B → B causes A)', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });

    kgLink(db, {
      sourceId: n1.nodeId!,
      targetId: n2.nodeId!,
      type: 'caused-by',
    });

    const edge = db.getEdge(n2.nodeId!, n1.nodeId!, 'causes');
    expect(edge).not.toBeNull();
  });
});

describe('kgNeighbors', () => {
  it('returns neighbors within depth', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });
    const n3 = kgAdd(db, { category: 'knowledge', content: 'C' });

    kgLink(db, { sourceId: n1.nodeId!, targetId: n2.nodeId!, type: 'relates-to' });
    kgLink(db, { sourceId: n2.nodeId!, targetId: n3.nodeId!, type: 'relates-to' });

    const result = kgNeighbors(db, n1.nodeId!, 2);
    expect(result.success).toBe(true);
    expect(result.neighbors).toHaveLength(2);
  });

  it('returns error for nonexistent node', () => {
    const result = kgNeighbors(db, 'nonexistent');
    expect(result.success).toBe(false);
  });
});

describe('kgGet', () => {
  it('returns node with edges', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });
    const n2 = kgAdd(db, { category: 'knowledge', content: 'B' });
    kgLink(db, { sourceId: n1.nodeId!, targetId: n2.nodeId!, type: 'blocks' });

    const result = kgGet(db, n1.nodeId!);
    expect(result.success).toBe(true);
    expect(result.node).toBeDefined();
    expect(result.edges).toHaveLength(1);
  });

  it('returns error for nonexistent node', () => {
    const result = kgGet(db, 'nonexistent');
    expect(result.success).toBe(false);
  });
});

describe('kgDelete', () => {
  it('deletes a node', () => {
    const n1 = kgAdd(db, { category: 'knowledge', content: 'A' });

    const result = kgDelete(db, n1.nodeId!);
    expect(result.success).toBe(true);
    expect(result.deleted).toBe(true);
  });

  it('returns error for nonexistent node', () => {
    const result = kgDelete(db, 'nonexistent');
    expect(result.success).toBe(false);
    expect(result.deleted).toBe(false);
  });
});

describe('kgQuery', () => {
  it('returns graph stats', () => {
    kgAdd(db, { category: 'knowledge', content: 'Test' });

    const result = kgQuery(db, {});
    expect(result.success).toBe(true);
    expect(result.stats.nodeCount).toBe(1);
  });

  it('uses queryType param (TL-3)', () => {
    kgAdd(db, { category: 'knowledge', content: 'Test' });

    const result = kgQuery(db, { queryType: 'search' });
    expect(result.success).toBe(true);
    expect(result.queryTypeDist).toBeDefined();
  });

  it('includes most surfaced and gaps', () => {
    const result = kgQuery(db, {});
    expect(result.mostSurfaced).toBeDefined();
    expect(result.gaps).toBeDefined();
    expect(result.distribution).toBeDefined();
  });
});
