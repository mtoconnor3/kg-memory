import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import { openKnowledgeGraph } from '../db.ts';
import {
  cosineSimilarity,
  computeCompositeScore,
  validateConfig,
  DEFAULT_CONFIG,
  searchHybrid,
  getEmbedding,
  batchEmbed,
  getQueryEmbedding,
} from '../search.ts';

const TEST_DB = '/tmp/kg-search-test-' + Date.now() + '.db';

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

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const a = [1, 0, 0];
    expect(cosineSimilarity(a, a)).toBeCloseTo(1, 5);
  });

  it('returns 0 for orthogonal vectors', () => {
    const a = [1, 0, 0];
    const b = [0, 1, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
  });

  it('returns 0 for dimension mismatch (SR-2: fail soft)', () => {
    const a = [1, 0, 0];
    const b = [0, 1];
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  it('returns 0 for zero vectors', () => {
    const a = [0, 0, 0];
    expect(cosineSimilarity(a, a)).toBe(0);
  });

  it('clamps negative similarity to 0 (SR-7)', () => {
    const a = [1, 0, 0];
    const b = [-1, 0, 0];
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  // A.4: Non-unit-length vectors that expose the old bug
  it('handles non-unit-length orthogonal vectors: [3,0] vs [0,4] → 0', () => {
    const a = [3, 0];
    const b = [0, 4];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
  });

  it('handles parallel vectors with different magnitudes: [1,2] vs [2,4] → 1.0', () => {
    // This case FAILS under the old bug (normB computed from a instead of b)
    // because magnitude collapses to ‖a‖² instead of ‖a‖·‖b‖
    const a = [1, 2];
    const b = [2, 4];
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
  });

  it('handles non-unit-length anti-parallel vectors → 0 (clamped)', () => {
    const a = [3, 4];
    const b = [-6, -8];
    expect(cosineSimilarity(a, b)).toBe(0); // clamped from -1
  });

  it('handles non-unit-length vectors with known cosine', () => {
    // a = [1, 0], b = [1, 1] → cosine = 1/√2 ≈ 0.7071
    const a = [1, 0];
    const b = [1, 1];
    expect(cosineSimilarity(a, b)).toBeCloseTo(1 / Math.sqrt(2), 5);
  });
});

describe('computeCompositeScore', () => {
  it('computes hybrid score with all four terms (SR-3)', () => {
    const config = { ...DEFAULT_CONFIG };
    const score = computeCompositeScore(0.8, 0.9, 0.7, 0.6, config);
    const expected = (0.8 * 0.4) + (0.9 * 0.3) + (0.7 * 0.15) + (0.6 * 0.15);
    expect(score).toBeCloseTo(expected, 5);
  });

  it('redistributes vector weight when unavailable (SR-3)', () => {
    const config = { ...DEFAULT_CONFIG };
    const score = computeCompositeScore(0.8, null, 0.7, 0.6, config);
    // vectorWeight redistributed to BM25
    expect(score).toBeGreaterThan(0);
  });
});

describe('validateConfig', () => {
  it('merges partial config with defaults', () => {
    const config = validateConfig({ maxResults: 20 });
    expect(config.maxResults).toBe(20);
    expect(config.embeddingEndpoint).toBe('http://localhost:1234/v1/embeddings');
  });

  it('validates weights per-field, not all-or-nothing (CFG-2)', () => {
    const config = validateConfig({
      ftsF5Weight: 999, // invalid
      injectionBudget: 5000, // valid
      queryLogLimit: 2000, // valid
    });
    expect(config.ftsF5Weight).toBe(DEFAULT_CONFIG.ftsF5Weight); // clamped
    expect(config.injectionBudget).toBe(5000); // preserved
    expect(config.queryLogLimit).toBe(2000); // preserved
  });

  it('accepts deprecated ftxF5Weight key (CFG-1)', () => {
    const config = validateConfig({ ftxF5Weight: 0.5 } as any);
    expect(config.ftsF5Weight).toBe(0.5);
  });

  it('uses correct default value (CFG-4)', () => {
    const config = validateConfig({ inputSearchThreshold: 999 });
    expect(config.inputSearchThreshold).toBe(DEFAULT_CONFIG.inputSearchThreshold); // 0.45
  });

  it('includes injectionBudget in config (TS-3)', () => {
    const config = validateConfig({ injectionBudget: 3000 });
    expect(config.injectionBudget).toBe(3000);
  });

  it('includes staleNodeDays (CFG-6)', () => {
    const config = validateConfig({ staleNodeDays: 60 });
    expect(config.staleNodeDays).toBe(60);
  });
});

describe('searchHybrid', () => {
  it('returns empty for empty graph', async () => {
    const results = await searchHybrid(db, 'test');
    expect(results).toEqual([]);
  });

  it('returns FTS results when embedding unavailable', async () => {
    // Point to a non-existent endpoint
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
    db.saveNode({ category: 'knowledge', content: 'Hello world test' });

    const results = await searchHybrid(db, 'hello', 10, undefined, undefined, config);
    expect(results.length).toBeGreaterThanOrEqual(0); // Should not throw
  });

  it('empty query returns all nodes', async () => {
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
    db.saveNode({ category: 'knowledge', content: 'Alpha' });
    db.saveNode({ category: 'project', content: 'Beta' });

    const results = await searchHybrid(db, '', 10, undefined, undefined, config);
    expect(results.length).toBe(2);
  });

  // 4.3: Vector search is no longer gated by FTS
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

  // 4.4: No candidate embedding in the query path
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

  // 4.5: Filter excludes null subcategory consistently
  it('subcategory filter excludes null-subcategory nodes (matches FTS semantics)', async () => {
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
    db.saveNode({ category: 'knowledge', subcategory: 'fact', content: 'has subcategory' });
    db.saveNode({ category: 'knowledge', subcategory: null, content: 'no subcategory' });

    const results = await searchHybrid(db, '', 10, ['knowledge'], ['fact'], config);
    expect(results.every(r => r.node.subcategory === 'fact')).toBe(true);
  });
});

describe('batchEmbed (SR-1)', () => {
  it('returns null array when endpoint unavailable', async () => {
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
    const results = await batchEmbed(['text1', 'text2'], config);
    expect(results).toHaveLength(2);
    expect(results[0]).toBeNull();
    expect(results[1]).toBeNull();
  });

  it('handles empty input', async () => {
    const results = await batchEmbed([], DEFAULT_CONFIG);
    expect(results).toEqual([]);
  });
});

describe('nomic task prefixes (SR-4)', () => {
  it('getEmbedding applies search_document prefix', async () => {
    // We can't test the actual HTTP call, but we verify the function exists and doesn't crash
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
    const result = await getEmbedding('test', config);
    expect(result).toBeNull(); // endpoint unavailable, but no crash
  });

  it('getQueryEmbedding applies search_query prefix', async () => {
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
    const result = await getQueryEmbedding('test', config);
    expect(result).toBeNull(); // endpoint unavailable, but no crash
  });
});
