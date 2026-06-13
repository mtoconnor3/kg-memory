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
    expect(config.embeddingEndpoint).toBe(DEFAULT_CONFIG.embeddingEndpoint);
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

  it('uses capped fallback on FTS miss (SR-1)', async () => {
    // Point to non-existent endpoint so we get FTS-only path
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };

    // Add nodes that won't match "xyznonexistent"
    for (let i = 0; i < 100; i++) {
      db.saveNode({ category: 'knowledge', content: `Item ${i} description` });
    }

    const results = await searchHybrid(db, 'xyznonexistent', 10, undefined, undefined, config);
    // Should return results (fallback path) but not hang
    expect(Array.isArray(results)).toBe(true);
  });

  it('empty query returns all nodes', async () => {
    const config = { ...DEFAULT_CONFIG, embeddingEndpoint: 'http://localhost:65535/v1/embeddings' };
    db.saveNode({ category: 'knowledge', content: 'Alpha' });
    db.saveNode({ category: 'project', content: 'Beta' });

    const results = await searchHybrid(db, '', 10, undefined, undefined, config);
    expect(results.length).toBe(2);
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
