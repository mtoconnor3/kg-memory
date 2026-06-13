/**
 * Knowledge Graph — Query Log Management
 *
 * Manages the rolling query log window and provides analytics queries:
 *   logQuery, pruneOldEntries, getMostSurfacedNodes, getZeroResultQueries,
 *   getGraphGrowth, getAgentActionDistribution, getCategoryDistribution
 */

import type { KnowledgeGraphDB } from './db.ts';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface LoggingConfig {
  queryLogLimit: number;
}

const DEFAULT_LOGGING_CONFIG: LoggingConfig = {
  queryLogLimit: 1000,
};

// ---------------------------------------------------------------------------
// Query log management
// ---------------------------------------------------------------------------

/**
 * Log a query operation to the query log.
 * This is a convenience wrapper around db.logQuery().
 */
export function logQuery(
  db: KnowledgeGraphDB,
  entry: {
    query: string;
    queryType: string;
    resultsReturned: number;
    relevanceScore?: number;
    injectedIds?: string[];
    injectedTokenBudget?: number;
    agentAction?: string;
  },
): void {
  db.logQuery(entry);
}

/**
 * Prune old query log entries, keeping the most recent N.
 */
export function pruneOldEntries(
  db: KnowledgeGraphDB,
  limit: number = DEFAULT_LOGGING_CONFIG.queryLogLimit,
): void {
  db.pruneQueryLog(limit);
}

// ---------------------------------------------------------------------------
// Analytics queries
// ---------------------------------------------------------------------------

/**
 * Most frequently surfaced nodes (by query log hits).
 * Returns the top N nodes that have been returned most often in search results.
 */
export function getMostSurfacedNodes(
  db: KnowledgeGraphDB,
  limit: number = 20,
): Array<{ nodeId: string; hits: number }> {
  // Query the query log for search operations that returned results
  const rows = db.db.prepare(`
    SELECT injected_ids, results_returned
    FROM query_log
    WHERE query_type = 'search' AND injected_ids IS NOT NULL AND results_returned > 0
  `).all() as Array<{ injected_ids: string; results_returned: number }>;

  // Parse injected_ids (JSON array) and aggregate hits
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
 * Returns queries that returned 0 results, indicating missing knowledge.
 */
export function getZeroResultQueries(
  db: KnowledgeGraphDB,
  limit: number = 20,
): Array<{ query: string; count: number }> {
  return db.getZeroResultQueries(limit);
}

/**
 * Graph growth over time.
 * Returns the number of nodes created per day.
 */
export function getGraphGrowth(
  db: KnowledgeGraphDB,
  dbPath: string,
  limit: number = 30,
): Array<{ date: string; nodeCount: number }> {
  // Get nodes grouped by creation date (created_at is unix ms, convert to seconds)
  const rows = db.db.prepare(`
    SELECT DATE(created_at / 1000, 'unixepoch') AS date, COUNT(*) AS nodeCount
    FROM nodes
    GROUP BY date
    ORDER BY date DESC
    LIMIT ?
  `).all(limit) as Array<{ date: string; nodeCount: number }>;

  // Calculate cumulative count
  let cumulative = 0;
  return rows.reverse().map(row => {
    cumulative += row.nodeCount;
    return { date: row.date, nodeCount: cumulative };
  });
}

/**
 * Agent action distribution.
 * Returns how many queries were used, ignored, or acted-on.
 */
export function getAgentActionDistribution(
  db: KnowledgeGraphDB,
): Array<{ action: string; count: number }> {
  const rows = db.db.prepare(`
    SELECT agent_action, COUNT(*) AS count
    FROM query_log
    WHERE agent_action IS NOT NULL
    GROUP BY agent_action
    ORDER BY count DESC
  `).all() as Array<{ action: string; count: number }>;

  return rows;
}

/**
 * Category distribution of nodes.
 * Returns the number of nodes per category/subcategory.
 */
export function getCategoryDistribution(
  db: KnowledgeGraphDB,
): Array<{ category: string; subcategory: string; count: number }> {
  return db.getCategoryDistribution();
}

/**
 * Query type distribution.
 * Returns the number of operations per type (search, kg_add, kg_link, etc.).
 */
export function getQueryTypeDistribution(
  db: KnowledgeGraphDB,
): Array<{ queryType: string; count: number }> {
  const rows = db.db.prepare(`
    SELECT query_type, COUNT(*) AS count
    FROM query_log
    GROUP BY query_type
    ORDER BY count DESC
  `).all() as Array<{ queryType: string; count: number }>;

  return rows;
}

// ---------------------------------------------------------------------------
// Full analytics report
// ---------------------------------------------------------------------------

export interface AnalyticsReport {
  graphStats: Record<string, any>;
  mostSurfaced: Array<{ nodeId: string; hits: number }>;
  gaps: Array<{ query: string; count: number }>;
  distribution: Array<{ category: string; subcategory: string; count: number }>;
  queryTypeDistribution: Array<{ query_type: string; count: number }>;
  agentActions: Array<{ agent_action: string; count: number }>;
  growth: Array<{ date: string; nodeCount: number }>;
}

/**
 * Generate a full analytics report.
 */
export function generateAnalyticsReport(
  db: KnowledgeGraphDB,
  dbPath: string,
  config: LoggingConfig = DEFAULT_LOGGING_CONFIG,
): AnalyticsReport {
  return {
    graphStats: db.getGraphStats(),
    mostSurfaced: getMostSurfacedNodes(db, 20),
    gaps: getZeroResultQueries(db, 20),
    distribution: getCategoryDistribution(db),
    queryTypeDistribution: getQueryTypeDistribution(db),
    agentActions: getAgentActionDistribution(db),
    growth: getGraphGrowth(db, dbPath, 30),
  };
}

/**
 * Generate a formatted analytics report string.
 */
export function generateFormattedReport(
  db: KnowledgeGraphDB,
  dbPath: string,
  config: LoggingConfig = DEFAULT_LOGGING_CONFIG,
): string {
  const report = generateAnalyticsReport(db, dbPath, config);
  return formatAnalyticsReport(report, db);
}

/**
 * Format an analytics report as a human-readable string.
 */
export function formatAnalyticsReport(report: AnalyticsReport, db?: KnowledgeGraphDB): string {
  const lines: string[] = [];

  // Graph stats
  const stats = report.graphStats;
  lines.push(`Knowledge Graph Analytics`);
  lines.push(`========================`);
  lines.push(`Nodes: ${stats.nodeCount} | Edges: ${stats.edgeCount} | Log entries: ${stats.queryLogSize}`);
  lines.push('');

  // Category distribution
  if (report.distribution.length > 0) {
    lines.push('Category Distribution:');
    for (const d of report.distribution) {
      lines.push(`  ${d.category}${d.subcategory ? `/${d.subcategory}` : ''}: ${d.count}`);
    }
    lines.push('');
  }

  // Most surfaced nodes
  if (report.mostSurfaced.length > 0 && db) {
    lines.push('Most Surfaced Nodes:');
    for (const m of report.mostSurfaced.slice(0, 10)) {
      const node = db.getNode(m.nodeId);
      if (node) {
        lines.push(`  ${m.hits} hits: [${node.category}${node.subcategory ? '/' + node.subcategory : ''}] ${node.content.slice(0, 80)}`);
      }
    }
    lines.push('');
  }

  // Gaps (zero-result queries)
  if (report.gaps.length > 0) {
    lines.push('Gaps (zero-result queries):');
    for (const g of report.gaps.slice(0, 10)) {
      lines.push(`  "${g.query}" (${g.count} attempts)`);
    }
    lines.push('');
  }

  // Query type distribution
  if (report.queryTypeDistribution.length > 0) {
    lines.push('Query Types:');
    for (const q of report.queryTypeDistribution) {
      lines.push(`  ${q.query_type}: ${q.count}`);
    }
    lines.push('');
  }

  // Agent actions
  if (report.agentActions.length > 0) {
    lines.push('Agent Actions:');
    for (const a of report.agentActions) {
      lines.push(`  ${a.agent_action}: ${a.count}`);
    }
    lines.push('');
  }

  // Growth (last 7 days)
  const recentGrowth = report.growth.slice(-7);
  if (recentGrowth.length > 0) {
    lines.push('Recent Growth (last 7 days):');
    for (const g of recentGrowth) {
      lines.push(`  ${g.date}: ${g.nodeCount} total nodes`);
    }
  }

  return lines.join('\n');
}
