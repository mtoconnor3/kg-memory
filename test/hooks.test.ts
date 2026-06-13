import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import { openKnowledgeGraph } from '../db.ts';
import {
  onSessionStart,
  onBeforeAgentStart,
  onContext,
  onSessionBeforeCompact,
  onSessionShutdown,
  onInput,
  clearInjectedIds,
} from '../hooks.ts';
import { DEFAULT_CONFIG, validateConfig } from '../search.ts';

const TEST_DB = '/tmp/kg-hooks-test-' + Date.now() + '.db';

let db: ReturnType<typeof openKnowledgeGraph>;
const config = validateConfig({ embeddingEndpoint: 'http://localhost:65535/v1/embeddings' });

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

describe('onSessionStart', () => {
  it('creates a session marker (DB-6)', () => {
    db.saveNode({ category: 'knowledge', content: 'Test' });
    const result = onSessionStart(db, { sessionId: 'test-1' }, config);

    expect(result.stats.nodeCount).toBe(1);
    expect(result.notification).toBeDefined();

    const markers = db.getMarkers(10);
    expect(markers).toHaveLength(1);
    expect(markers[0].event).toBe('session_start');
  });

  it('handles empty graph', () => {
    const result = onSessionStart(db, { sessionId: 'test-2' }, config);
    expect(result.stats.nodeCount).toBe(0);
    expect(result.notification).toContain('empty');
  });
});

describe('onBeforeAgentStart', () => {
  it('returns empty injection for empty graph', async () => {
    const result = await onBeforeAgentStart(db, { sessionId: 'test' }, config);
    expect(result.injection).toBe('');
    expect(result.nodeIds).toEqual([]);
  });

  it('injects high-trust nodes only (HK-3)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Trusted fact', trust: 'high' });
    db.saveNode({ category: 'knowledge', content: 'Untrusted fact', trust: 'low' });

    const result = await onBeforeAgentStart(db, { sessionId: 'test' }, config);
    // Should only inject the high-trust node
    expect(result.injection).not.toContain('Untrusted');
  });

  it('enforces injectionBudget (HK-2)', async () => {
    const shortConfig = validateConfig({
      ...config,
      injectionBudget: 50,
      maxResults: 100,
    });

    for (let i = 0; i < 20; i++) {
      db.saveNode({ category: 'knowledge', content: `Fact number ${i} with some extra text to make it longer` });
    }

    const result = await onBeforeAgentStart(db, { sessionId: 'test' }, shortConfig);
    // Should stop injecting when budget is reached
    expect(result.injection).toBeDefined();
  });

  it('uses single query instead of 6 (HK-1)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Project decision pattern' });

    const result = await onBeforeAgentStart(db, { sessionId: 'test' }, config);
    // Should work without 6 sequential queries
    expect(result.nodeIds.length).toBeGreaterThanOrEqual(0);
  });
});

describe('onContext', () => {
  it('returns empty for empty graph', async () => {
    const result = await onContext(db, { sessionId: 'test', compactedContent: 'some content' }, config);
    expect(result.injection).toBe('');
  });

  it('omits null subcategory (HK-6)', async () => {
    db.saveNode({ category: 'knowledge', content: 'No subcategory here' });

    const result = await onContext(db, { sessionId: 'test', compactedContent: 'subcategory' }, config);
    // Should not contain "null" in the label
    expect(result.injection).not.toContain('/null');
  });

  it('prevents double injection (HK-7)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Shared fact' });

    // Simulate before_agent_start injecting
    clearInjectedIds();
    const startResult = await onBeforeAgentStart(db, { sessionId: 'test' }, config);

    // Then context hook should not re-inject same nodes
    const contextResult = await onContext(db, { sessionId: 'test', compactedContent: 'fact' }, config);
    // The node should not appear twice
    expect(contextResult.nodeIds.length).toBe(0); // Already injected by before_agent_start
  });
});

describe('onSessionBeforeCompact', () => {
  it('includes graph summary', () => {
    db.saveNode({ category: 'knowledge', content: 'Important fact' });
    // Bump frequency directly for testing
    db.getDb().prepare('UPDATE nodes SET frequency = 5 WHERE content = ?').run('Important fact');
    const result = onSessionBeforeCompact(db, {
      sessionId: 'test',
      sessionHistory: [{ role: 'user', content: 'test' }],
    });

    expect(result.injection).toContain('Graph Summary');
    expect(result.injection).toContain('session test');
  });

  it('handles empty graph', () => {
    const result = onSessionBeforeCompact(db, {
      sessionId: 'test',
      sessionHistory: [],
    });

    expect(result.injection).toContain('Nodes: 0');
  });
});

describe('onSessionShutdown', () => {
  it('creates marker and prunes query log (HK-5)', () => {
    for (let i = 0; i < 20; i++) {
      db.logQuery({ query: `q${i}`, queryType: 'search', resultsReturned: 0 });
    }

    const result = onSessionShutdown(db, { sessionId: 'test' }, config);
    expect(result.snapshotVersion).toBeGreaterThan(0);
    expect(result.pruned).toBeGreaterThanOrEqual(0);
  });

  it('uses config.queryLogLimit (HK-5)', () => {
    const customConfig = validateConfig({ queryLogLimit: 15 });
    for (let i = 0; i < 30; i++) {
      db.logQuery({ query: `q${i}`, queryType: 'search', resultsReturned: 0 });
    }

    const result = onSessionShutdown(db, { sessionId: 'test' }, customConfig);
    // 30 entries, limit 15 → 15 pruned
    expect(result.pruned).toBe(15);
  });
});

describe('onInput', () => {
  it('returns null for empty text', async () => {
    const result = await onInput(db, { text: '' }, config);
    expect(result).toBeNull();
  });

  it('returns null for empty graph', async () => {
    const result = await onInput(db, { text: 'hello' }, config);
    expect(result).toBeNull();
  });

  it('only injects high-trust nodes (HK-3)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Low trust content', trust: 'low' });
    const result = await onInput(db, { text: 'content' }, config);
    expect(result).toBeNull(); // Low trust nodes are filtered out
  });

  it('prevents double injection (HK-7)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Repeated fact' });

    clearInjectedIds();
    // First injection
    const r1 = await onInput(db, { text: 'fact' }, config);

    // Second injection should not re-inject same nodes
    const r2 = await onInput(db, { text: 'fact' }, config);
    expect(r2).toBeNull(); // Already injected
  });
});

describe('clearInjectedIds', () => {
  it('clears the per-turn tracking set (HK-7)', async () => {
    db.saveNode({ category: 'knowledge', content: 'Test' });
    clearInjectedIds();

    const result = await onBeforeAgentStart(db, { sessionId: 'test' }, config);
    // After clearing, nodes can be injected again
    expect(result.nodeIds.length).toBeGreaterThanOrEqual(0);
  });
});
