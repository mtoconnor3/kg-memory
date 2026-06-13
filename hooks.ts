/**
 * Knowledge Graph — Session Lifecycle Hooks
 *
 * Integrates the knowledge graph with Pi's session system:
 *   session_start, before_agent_start, context, session_before_compact, session_shutdown
 */

import type { KnowledgeGraphDB, SearchHit } from './db.ts';
import type { SearchConfig } from './search.ts';
import { DEFAULT_CONFIG, searchHybrid } from './search.ts';

// ---------------------------------------------------------------------------
// Hook interfaces
// ---------------------------------------------------------------------------

export interface SessionStartContext {
  sessionId: string;
  project?: string;
}

export interface BeforeAgentStartContext {
  sessionId: string;
  previousSessionId?: string;
}

export interface ContextHookContext {
  sessionId: string;
  compactedContent: string;
}

export interface SessionBeforeCompactContext {
  sessionId: string;
  sessionHistory: Array<{ role: string; content: string }>;
}

export interface SessionShutdownContext {
  sessionId: string;
}

export interface InputContext {
  text: string;
  images?: Array<{ type: string; source: any }>;
  source?: 'interactive' | 'rpc' | 'extension';
  streamingBehavior?: 'steer' | 'followUp' | undefined;
}

// ---------------------------------------------------------------------------
// Hook implementations
// ---------------------------------------------------------------------------

/**
 * session_start — Log graph stats and notify user.
 */
export function onSessionStart(
  db: KnowledgeGraphDB,
  context: SessionStartContext,
  config: SearchConfig = DEFAULT_CONFIG,
): { stats: Record<string, any>; notification?: string } {
  const stats = db.getGraphStats();

  // Create a snapshot for versioning
  db.createSnapshot(`Session ${context.sessionId} started`);

  // Build notification message (if UI available)
  let notification: string | undefined;
  if (stats.nodeCount > 0) {
    const catSummary = Object.entries(stats.categoryDistribution)
      .filter(([, count]) => count > 0)
      .map(([cat, count]) => `${count} ${cat}`)
      .join(', ');

    notification = `Knowledge graph: ${stats.nodeCount} nodes, ${stats.edgeCount} edges (${catSummary})`;
  } else {
    notification = 'Knowledge graph: empty — start adding nodes with kg_add()';
  }

  return { stats, notification };
}

/**
 * before_agent_start — Inject top-N relevant facts into the system prompt.
 *
 * At session start, query the graph for likely-relevant facts and format
 * them as a concise summary (< injectionBudget tokens).
 */
export async function onBeforeAgentStart(
  db: KnowledgeGraphDB,
  context: BeforeAgentStartContext,
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<{ injection: string; nodeIds: string[] }> {
  // Early exit: skip 6 LMStudio calls when the graph is empty.
  if (db.getNodeCount() === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Search for broadly relevant facts using multiple queries
  // (FTS5 uses AND by default, so we search keywords separately and merge)
  const keywords = ['project', 'decision', 'pattern', 'preference', 'warning', 'fact'];
  const allResults: SearchHit[] = [];
  const seenIds = new Set<string>();

  for (const keyword of keywords) {
    const results = await searchHybrid(db, keyword, config.maxResults);
    for (const hit of results) {
      if (!seenIds.has(hit.node.id)) {
        seenIds.add(hit.node.id);
        allResults.push(hit);
      }
    }
  }

  // Sort by score and take top-N
  allResults.sort((a, b) => b.compositeScore - a.compositeScore);
  const topResults = allResults.slice(0, config.maxResults);

  if (topResults.length === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Format as a concise summary
  const injectionParts: string[] = [];
  let tokenBudget = config.injectionBudget;

  for (const hit of topResults) {
    const line = formatNodeSummary(hit.node);
    if (line.length >= tokenBudget) break;

    injectionParts.push(line);
    tokenBudget -= line.length;
  }

  const injection = injectionParts.length > 0
    ? `[Knowledge Graph]\n  ${injectionParts.join('\n  ')}\n[End Knowledge Graph]`
    : '';

  return {
    injection,
    nodeIds: topResults.map(r => r.node.id),
  };
}

/**
 * context — After compaction, inject pointers to graph nodes
 * that appear in the compacted region.
 */
export async function onContext(
  db: KnowledgeGraphDB,
  context: ContextHookContext,
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<{ injection: string; nodeIds: string[] }> {
  // Early exit: skip LMStudio call when the graph is empty.
  if (db.getNodeCount() === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Use hybrid search instead of pure FTS5
  const results = await searchHybrid(db, context.compactedContent, 5, undefined, undefined, config);

  if (results.length === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Format as pointers (full details available via kg_get)
  const injection = results.map(r =>
    `* [${r.node.category}/${r.node.subcategory}] ${r.node.content.slice(0, 100)}`,
  ).join('\n');

  return {
    injection: `[Knowledge Graph Pointers]\n  ${injection}\n[End Knowledge Graph Pointers]`,
    nodeIds: results.map(r => r.node.id),
  };
}

/**
 * session_before_compact — Inject graph summary into the compaction prompt.
 *
 * This preserves graph knowledge across compaction by including a summary
 * of the entire graph in the compaction prompt.
 */
export function onSessionBeforeCompact(
  db: KnowledgeGraphDB,
  context: SessionBeforeCompactContext,
): { injection: string } {
  const stats = db.getGraphStats();

  // Build a summary of the graph
  const lines: string[] = [];
  lines.push(`Graph Summary (session ${context.sessionId}):`);
  lines.push(`  Nodes: ${stats.nodeCount}, Edges: ${stats.edgeCount}`);

  for (const [cat, count] of Object.entries(stats.categoryDistribution)) {
    if (count > 0) {
      lines.push(`  ${cat}: ${count}`);
    }
  }

  // Top nodes by frequency (most important facts)
  const topNodes = db.db.prepare(`
    SELECT id, content, category, subcategory, frequency
    FROM nodes ORDER BY frequency DESC LIMIT 5
  `).all() as Array<{ id: string; content: string; category: string; subcategory: string | null; frequency: number }>;

  if (topNodes.length > 0) {
    lines.push('  Top facts:');
    for (const n of topNodes) {
      lines.push(`    - [${n.category}${n.subcategory ? '/' + n.subcategory : ''}] ${n.content.slice(0, 120)}`);
    }
  }

  return {
    injection: `[Graph Context]\n${lines.join('\n')}\n[End Graph Context]`,
  };
}

/**
 * session_shutdown — Save graph state, log snapshot, prune old query log.
 */
export function onSessionShutdown(
  db: KnowledgeGraphDB,
  context: SessionShutdownContext,
): { snapshotVersion: number; pruned: number } {
  // Create a snapshot
  const version = db.createSnapshot(`Session ${context.sessionId} ended`);

  // Prune old query log entries
  db.pruneQueryLog(1000);

  return { snapshotVersion: version, pruned: 0 };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Format a node as a concise summary line for system prompt injection.
 */
function formatNodeSummary(node: {
  id: string;
  category: string;
  subcategory: string | null;
  content: string;
}): string {
  const label = node.subcategory
    ? `${node.category}/${node.subcategory}`
    : node.category;
  return `- ${node.content.slice(0, 200)} (${label})`;
}

/**
 * Format high-scoring search results as an injection block.
 * Each result gets a line: [category/subcategory] Content (score: X.XX)
 */
function formatInputSurface(results: SearchHit[]): string {
  const lines = results.map(r => {
    const label = r.node.subcategory
      ? `${r.node.category}/${r.node.subcategory}`
      : r.node.category;
    const score = r.compositeScore.toFixed(3);
    const content = r.node.content.slice(0, 200);
    return `  - [${label}] ${content} (score: ${score})`;
  });

  return `[KG Auto-Surface]\n  ${lines.join('\n  ')}\n[End KG Auto-Surface]`;
}

/**
 * onInput — Fires when user submits input.
 *
 * Searches the knowledge graph for relevant nodes. If any nodes
 * score >= inputSearchThreshold, formats them as an injection block
 * and returns it for system prompt injection.
 *
 * Returns null if no results meet the threshold (no injection).
 */
export async function onInput(
  db: KnowledgeGraphDB,
  event: InputContext,
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<string | null> {
  const text = (event.text || '').trim();
  if (!text) return null;

  // Early exit: skip LMStudio call when the graph is empty.
  if (db.getNodeCount() === 0) {
    return null;
  }

  // Run hybrid search with input-specific config
  const results = await searchHybrid(
    db,
    text,
    config.inputSearchMaxResults,
    undefined,    // no category filter
    undefined,    // no subcategory filter
    config,
  );

  if (results.length === 0) return null;

  // Filter to high-scoring results
  const highScore = results.filter(r => r.compositeScore >= config.inputSearchThreshold);
  if (highScore.length === 0) return null;

  // Format as injection block
  return formatInputSurface(highScore);
}
