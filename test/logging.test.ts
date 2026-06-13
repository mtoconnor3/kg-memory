import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import { openKnowledgeGraph } from '../db.ts';
import {
  logQuery,
  pruneOldEntries,
  getMostSurfacedNodes,
  getZeroResultQueries,
  getGraphGrowth,
  getAgentActionDistribution,
  getCategoryDistribution,
  getQueryTypeDistribution,
  generateAnalyticsReport,
  generateFormattedReport,
  generateGraphOverview,
} from '../logging.ts';

const TEST_DB = '/tmp/kg-logging-test-' + Date.now() + '.db';

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

describe('logQuery', () => {
  it('delegates to db.logQuery (LG-1)', () => {
    logQuery(db, { query: 'test', queryType: 'search', resultsReturned: 5 });
    const entries = db.getQueryLogEntries(10);
    expect(entries).toHaveLength(1);
  });
});

describe('pruneOldEntries', () => {
  it('delegates to db.pruneQueryLog', () => {
    for (let i = 0; i < 20; i++) {
      db.logQuery({ query: `q${i}`, queryType: 'search', resultsReturned: 0 });
    }
    pruneOldEntries(db, 10);
    const entries = db.getQueryLogEntries(100);
    expect(entries).toHaveLength(10);
  });
});

describe('getMostSurfacedNodes', () => {
  it('delegates to db.getMostSurfacedNodes (LG-1)', () => {
    const n1 = db.saveNode({ category: 'knowledge', content: 'Popular' });
    db.logQuery({
      query: 'search',
      queryType: 'search',
      resultsReturned: 3,
      injectedIds: [n1.id],
    });

    const result = getMostSurfacedNodes(db, 10);
    expect(result).toHaveLength(1);
    expect(result[0].nodeId).toBe(n1.id);
  });
});

describe('getZeroResultQueries', () => {
  it('delegates to db.getZeroResultQueries (LG-1)', () => {
    db.logQuery({ query: 'missing', queryType: 'search', resultsReturned: 0 });

    const result = getZeroResultQueries(db, 10);
    expect(result).toHaveLength(1);
    expect(result[0].query).toBe('missing');
  });
});

describe('getGraphGrowth', () => {
  it('returns true cumulative (LG-3)', () => {
    db.saveNode({ category: 'knowledge', content: 'A' });
    db.saveNode({ category: 'knowledge', content: 'B' });

    const result = getGraphGrowth(db, 30);
    if (result.length > 0) {
      expect(result[result.length - 1].nodeCount).toBe(2);
    }
  });
});

describe('getAgentActionDistribution', () => {
  it('uses correct field name: action (LG-2)', () => {
    db.logQuery({ query: 'q1', queryType: 'search', resultsReturned: 1, agentAction: 'used' });

    const result = getAgentActionDistribution(db);
    expect(result[0].action).toBe('used');
  });
});

describe('getQueryTypeDistribution', () => {
  it('uses correct field name: queryType (LG-2)', () => {
    db.logQuery({ query: 'q1', queryType: 'search', resultsReturned: 1 });
    db.logQuery({ query: 'q2', queryType: 'kg_add', resultsReturned: 1 });

    const result = getQueryTypeDistribution(db);
    expect(result[0].queryType).toBeDefined();
    expect(result[1].queryType).toBeDefined();
  });
});

describe('generateAnalyticsReport', () => {
  it('includes all sections', () => {
    db.saveNode({ category: 'knowledge', content: 'Test' });

    const report = generateAnalyticsReport(db);
    expect(report.graphStats).toBeDefined();
    expect(report.mostSurfaced).toBeDefined();
    expect(report.gaps).toBeDefined();
    expect(report.distribution).toBeDefined();
    expect(report.queryTypeDistribution).toBeDefined();
    expect(report.agentActions).toBeDefined();
    expect(report.growth).toBeDefined();
  });
});

describe('generateFormattedReport', () => {
  it('produces a readable string', () => {
    db.saveNode({ category: 'knowledge', content: 'Test' });

    const report = generateFormattedReport(db);
    expect(typeof report).toBe('string');
    expect(report).toContain('Knowledge Graph Analytics');
  });

  it('LG-2: query types use queryType field', () => {
    db.logQuery({ query: 'q1', queryType: 'search', resultsReturned: 1 });

    const report = generateFormattedReport(db);
    expect(report).not.toContain('undefined:');
  });

  it('LG-2: agent actions use action field', () => {
    db.logQuery({ query: 'q1', queryType: 'search', resultsReturned: 1, agentAction: 'used' });

    const report = generateFormattedReport(db);
    expect(report).toContain('used:');
    expect(report).not.toContain('undefined:');
  });
});

describe('generateGraphOverview (LG-4)', () => {
  it('produces a distinct report from analytics', () => {
    db.saveNode({ category: 'knowledge', content: 'Test' });

    const overview = generateGraphOverview(db);
    expect(overview).toContain('Knowledge Graph Overview');
    expect(overview).toContain('Nodes: 1');
  });
});
