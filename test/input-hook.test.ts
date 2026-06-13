import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import { openKnowledgeGraph } from '../db.ts';
import { onInput, clearInjectedIds } from '../hooks.ts';
import { DEFAULT_CONFIG, validateConfig } from '../search.ts';

const TEST_DB = '/tmp/kg-input-test-' + Date.now() + '.db';

let db: ReturnType<typeof openKnowledgeGraph>;
const config = validateConfig({
  embeddingEndpoint: 'http://localhost:65535/v1/embeddings',
  inputSearchTimeout: 2000,
  inputSearchThreshold: 0.3,
});

beforeEach(() => {
  db = openKnowledgeGraph(TEST_DB);
  clearInjectedIds();
});

afterEach(() => {
  db.close();
  try {
    fs.rmSync(TEST_DB, { force: true });
    fs.rmSync(TEST_DB + '-wal', { force: true });
    fs.rmSync(TEST_DB + '-shm', { force: true });
  } catch {}
});

describe('onInput', () => {
  it('returns null for empty text', async () => {
    const result = await onInput(db, { text: '' }, config);
    expect(result).toBeNull();
  });

  it('returns null for whitespace-only text', async () => {
    const result = await onInput(db, { text: '   ' }, config);
    expect(result).toBeNull();
  });

  it('returns null when graph is empty', async () => {
    const result = await onInput(db, { text: 'hello' }, config);
    expect(result).toBeNull();
  });

  it('returns null when LMStudio unavailable and no FTS match', async () => {
    db.saveNode({ category: 'knowledge', content: 'Some fact about dogs' });

    const result = await onInput(db, { text: 'xyznonexistent' }, config);
    // With no embedding and no FTS match, should return null
    expect(result).toBeNull();
  });

  it('returns injection for FTS match when LMStudio unavailable', async () => {
    db.saveNode({ category: 'knowledge', content: 'The project uses React' });

    const result = await onInput(db, { text: 'React' }, config);
    expect(result).not.toBeNull();
    expect(result).toContain('React');
  });

  it('only injects high-trust nodes (HK-3)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Low trust fact', trust: 'low' });

    const result = await onInput(db, { text: 'fact' }, config);
    expect(result).toBeNull();
  });

  it('prevents double injection within same turn (HK-7)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Important fact about React' });

    const r1 = await onInput(db, { text: 'React' }, config);
    const r2 = await onInput(db, { text: 'React' }, config);

    expect(r1).not.toBeNull();
    expect(r2).toBeNull(); // Already injected
  });

  it('clears injection tracking between turns', async () => {
    db.saveNode({ category: 'knowledge', content: 'Important fact about React' });

    // First turn
    const r1 = await onInput(db, { text: 'React' }, config);
    expect(r1).not.toBeNull();

    // Clear for next turn
    clearInjectedIds();

    // Second turn — should inject again
    const r2 = await onInput(db, { text: 'React' }, config);
    expect(r2).not.toBeNull();
  });

  it('respects inputSearchTimeout (HK-4)', async () => {
    const shortConfig = validateConfig({
      embeddingEndpoint: 'http://localhost:65535/v1/embeddings',
      inputSearchTimeout: 500,
    });

    db.saveNode({ category: 'knowledge', content: 'Timeout test' });

    const start = Date.now();
    const result = await onInput(db, { text: 'test' }, shortConfig);
    const elapsed = Date.now() - start;

    // Should not block for more than the configured timeout
    expect(elapsed).toBeLessThan(5000); // generous margin
  });
});
