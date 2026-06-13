/**
 * Knowledge Graph — Query Log Management & Analytics
 *
 * Manages the rolling query log window and provides analytics reports.
 * All SQL queries are in the db layer (LG-1) — this module only formats.
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
 * Delegates to db.logQuery() (LG-1: single source of truth).
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
// Analytics queries (delegated to db layer — LG-1)
// ---------------------------------------------------------------------------

/**
 * Most frequently surfaced nodes (by query log hits).
 * Delegates to db.getMostSurfacedNodes().
 */
export function getMostSurfacedNodes(
  db: KnowledgeGraphDB,
  limit: number = 20,
): Array<{ nodeId: string; hits: number }> {
  return db.getMostSurfacedNodes(limit);
}

/**
 * Zero-result queries (gaps in the graph).
 * Delegates to db.getZeroResultQueries().
 */
export function getZeroResultQueries(
  db: KnowledgeGraphDB,
  limit: number = 20,
): Array<{ query: string; count: number }> {
  return db.getZeroResultQueries(limit);
}

/**
 * Graph growth over time (LG-3: true cumulative).
 * Delegates to db.getGraphGrowth().
 */
export function getGraphGrowth(
  db: KnowledgeGraphDB,
  limit: number = 30,
): Array<{ date: string; nodeCount: number }> {
  return db.getGraphGrowth(limit);
}

/**
 * Agent action distribution (LG-2: aligned field names).
 * Delegates to db.getAgentActionDistribution().
 */
export function getAgentActionDistribution(
  db: KnowledgeGraphDB,
): Array<{ action: string; count: number }> {
  return db.getAgentActionDistribution();
}

/**
 * Category distribution of nodes.
 * Delegates to db.getCategoryDistribution().
 */
export function getCategoryDistribution(
  db: KnowledgeGraphDB,
): Array<{ category: string; subcategory: string; count: number }> {
  return db.getCategoryDistribution();
}

/**
 * Query type distribution (LG-2: aligned field names).
 * Delegates to db.getQueryTypeDistribution().
 */
export function getQueryTypeDistribution(
  db: KnowledgeGraphDB,
): Array<{ queryType: string; count: number }> {
  return db.getQueryTypeDistribution();
}

// ---------------------------------------------------------------------------
// Full analytics report
// ---------------------------------------------------------------------------

export interface AnalyticsReport {
  graphStats: Record<string, any>;
  mostSurfaced: Array<{ nodeId: string; hits: number }>;
  gaps: Array<{ query: string; count: number }>;
  distribution: Array<{ category: string; subcategory: string; count: number }>;
  queryTypeDistribution: Array<{ queryType: string; count: number }>;
  agentActions: Array<{ action: string; count: number }>;
  growth: Array<{ date: string; nodeCount: number }>;
}

/**
 * Generate a full analytics report.
 */
export function generateAnalyticsReport(
  db: KnowledgeGraphDB,
  config: LoggingConfig = DEFAULT_LOGGING_CONFIG,
): AnalyticsReport {
  return {
    graphStats: db.getGraphStats(),
    mostSurfaced: getMostSurfacedNodes(db, 20),
    gaps: getZeroResultQueries(db, 20),
    distribution: getCategoryDistribution(db),
    queryTypeDistribution: getQueryTypeDistribution(db),
    agentActions: getAgentActionDistribution(db),
    growth: getGraphGrowth(db, 30),
  };
}

/**
 * Generate a formatted graph overview (for /kg command — LG-4).
 */
export function generateGraphOverview(
  db: KnowledgeGraphDB,
): string {
  const stats = db.getGraphStats();
  const lines: string[] = [];

  lines.push('Knowledge Graph Overview');
  lines.push('========================');
  lines.push(`Nodes: ${stats.nodeCount} | Edges: ${stats.edgeCount} | Log entries: ${stats.queryLogSize}`);
  lines.push('');

  if (stats.oldestNode) {
    lines.push(`Oldest node: ${stats.oldestNode}`);
  }
  if (stats.newestNode) {
    lines.push(`Newest node: ${stats.newestNode}`);
  }
  lines.push('');

  // Category distribution
  if (stats.categoryDistribution) {
    lines.push('Categories:');
    for (const [cat, count] of Object.entries(stats.categoryDistribution)) {
      lines.push(`  ${cat}: ${count}`);
    }
    lines.push('');
  }

  // Most surfaced nodes
  const mostSurfaced = db.getMostSurfacedNodes(5);
  if (mostSurfaced.length > 0) {
    lines.push('Most Surfaced Nodes:');
    for (const m of mostSurfaced) {
      const node = db.getNode(m.nodeId);
      if (node) {
        const label = node.subcategory ? `${node.category}/${node.subcategory}` : node.category;
        lines.push(`  ${m.hits} hits: [${label}] ${node.content.slice(0, 80)}`);
      }
    }
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Generate a formatted analytics report (for /kg-query command — LG-4).
 */
export function generateFormattedReport(
  db: KnowledgeGraphDB,
  _dbPath: string = '',
  config: LoggingConfig = DEFAULT_LOGGING_CONFIG,
): string {
  const report = generateAnalyticsReport(db, config);
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
        const label = node.subcategory ? `${node.category}/${node.subcategory}` : node.category;
        lines.push(`  ${m.hits} hits: [${label}] ${node.content.slice(0, 80)}`);
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

  // Query type distribution (LG-2: aligned field names — queryType)
  if (report.queryTypeDistribution.length > 0) {
    lines.push('Query Types:');
    for (const q of report.queryTypeDistribution) {
      lines.push(`  ${q.queryType}: ${q.count}`);
    }
    lines.push('');
  }

  // Agent actions (LG-2: aligned field names — action)
  if (report.agentActions.length > 0) {
    lines.push('Agent Actions:');
    for (const a of report.agentActions) {
      lines.push(`  ${a.action}: ${a.count}`);
    }
    lines.push('');
  }

  // Growth (last 7 days) (LG-3: true cumulative)
  const recentGrowth = report.growth.slice(-7);
  if (recentGrowth.length > 0) {
    lines.push('Recent Growth (last 7 days):');
    for (const g of recentGrowth) {
      lines.push(`  ${g.date}: ${g.nodeCount} total nodes`);
    }
  }

  return lines.join('\n');
}
