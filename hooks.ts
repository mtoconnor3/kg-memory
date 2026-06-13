/**
 * Knowledge Graph — Session Lifecycle Hooks
 *
 * Integrates the knowledge graph with Pi's session system:
 *   session_start, before_agent_start, context, session_before_compact, session_shutdown
 */

import type { KnowledgeGraphDB, SearchHit } from './db.ts';
import type { KgConfig } from './search.ts';
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
// Per-turn injection tracking (HK-7: prevent double injection)
// ---------------------------------------------------------------------------

const injectedNodeIds = new Set<string>();

// ---------------------------------------------------------------------------
// Hook implementations
// ---------------------------------------------------------------------------

/**
 * session_start — Log graph stats and notify user.
 */
export function onSessionStart(
  db: KnowledgeGraphDB,
  context: SessionStartContext,
  config: KgConfig = DEFAULT_CONFIG,
): { stats: Record<string, any>; notification?: string } {
  const stats = db.getGraphStats();

  // Create a session marker (replaces snapshots — DB-6)
  db.createMarker('session_start', `Session ${context.sessionId} started`);

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
 * Uses a single broad search instead of 6 sequential queries (HK-1).
 * Enforces injectionBudget correctly (HK-2).
 * Only injects high-trust nodes (HK-3).
 * Tracks injected IDs to prevent double injection (HK-7).
 */
export async function onBeforeAgentStart(
  db: KnowledgeGraphDB,
  context: BeforeAgentStartContext,
  config: KgConfig = DEFAULT_CONFIG,
): Promise<{ injection: string; nodeIds: string[] }> {
  // Early exit: skip when graph is empty
  if (db.getNodeCount() === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Single broad search instead of 6 sequential queries (HK-1)
  const results = await searchHybrid(db, '', config.maxResults * 2);

  if (results.length === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Filter to high-trust nodes only (HK-3: provenance gating)
  const trustedResults = results.filter(r => r.node.trust === 'high');

  // Sort by score and take top-N
  trustedResults.sort((a, b) => b.compositeScore - a.compositeScore);
  const topResults = trustedResults.slice(0, config.maxResults);

  if (topResults.length === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Format with budget enforcement (HK-2: injectionBudget is now typed)
  const injectionParts: string[] = [];
  const injectedIds: string[] = [];
  let charBudget = config.injectionBudget;

  for (const hit of topResults) {
    const line = formatNodeSummary(hit.node);
    if (line.length >= charBudget) break;

    // Prevent double injection (HK-7)
    if (injectedNodeIds.has(hit.node.id)) continue;

    injectionParts.push(line);
    injectedIds.push(hit.node.id);
    injectedNodeIds.add(hit.node.id);
    charBudget -= line.length;
  }

  const injection = injectionParts.length > 0
    ? `[Knowledge Graph]\n  ${injectionParts.join('\n  ')}\n[End Knowledge Graph]`
    : '';

  return {
    injection,
    nodeIds: injectedIds,
  };
}

/**
 * context — After compaction, inject pointers to graph nodes
 * that appear in the compacted region.
 */
export async function onContext(
  db: KnowledgeGraphDB,
  context: ContextHookContext,
  config: KgConfig = DEFAULT_CONFIG,
): Promise<{ injection: string; nodeIds: string[] }> {
  // Early exit: skip when graph is empty
  if (db.getNodeCount() === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Use hybrid search with compacted content
  const results = await searchHybrid(db, context.compactedContent, 5, undefined, undefined, config);

  if (results.length === 0) {
    return { injection: '', nodeIds: [] };
  }

  // Format as pointers (HK-6: omit null subcategory)
  const injection = results
    .filter(r => !injectedNodeIds.has(r.node.id))  // HK-7: prevent double injection
    .map(r => {
      const label = r.node.subcategory
        ? `${r.node.category}/${r.node.subcategory}`
        : r.node.category;
      return `* [${label}] ${r.node.content.slice(0, 100)}`;
    })
    .join('\n');

  if (!injection) {
    return { injection: '', nodeIds: [] };
  }

  // Track injected IDs (HK-7)
  const newIds = results.filter(r => !injectedNodeIds.has(r.node.id)).map(r => r.node.id);
  newIds.forEach(id => injectedNodeIds.add(id));

  return {
    injection: `[Knowledge Graph Pointers]\n  ${injection}\n[End Knowledge Graph Pointers]`,
    nodeIds: newIds,
  };
}

/**
 * session_before_compact — Inject graph summary into the compaction prompt.
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
  const topNodes = db.getDb().prepare(`
    SELECT id, content, category, subcategory, frequency
    FROM nodes ORDER BY frequency DESC LIMIT 5
  `).all() as Array<{ id: string; content: string; category: string; subcategory: string | null; frequency: number }>;

  if (topNodes.length > 0) {
    lines.push('  Top facts:');
    for (const n of topNodes) {
      const label = n.subcategory ? `${n.category}/${n.subcategory}` : n.category;
      lines.push(`    - [${label}] ${n.content.slice(0, 120)}`);
    }
  }

  return {
    injection: `[Graph Context]\n${lines.join('\n')}\n[End Graph Context]`,
  };
}

/**
 * session_shutdown — Save graph state, prune query log, close DB (HK-5).
 */
export function onSessionShutdown(
  db: KnowledgeGraphDB,
  context: SessionShutdownContext,
  config: KgConfig = DEFAULT_CONFIG,
): { snapshotVersion: number; pruned: number } {
  // Create a session marker
  const markerId = db.createMarker('session_shutdown', `Session ${context.sessionId} ended`);

  // Prune old query log entries using config value (HK-5)
  const pruned = db.pruneQueryLog(config.queryLogLimit);

  // Prune old session markers (DB-6)
  db.pruneMarkers(50);

  // Close the database connection (HK-5)
  db.close();

  return { snapshotVersion: markerId, pruned };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Format a node as a concise summary line for system prompt injection.
 * (HK-6: omit null subcategory)
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
 * (HK-3: provenance-aware, delimited to prevent instruction injection)
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
 * score >= inputSearchThreshold, formats them as an injection block.
 *
 * (HK-4: uses configured timeout, non-blocking)
 * (HK-3: only injects high-trust nodes)
 * (HK-7: prevents double injection)
 */
export async function onInput(
  db: KnowledgeGraphDB,
  event: InputContext,
  config: KgConfig = DEFAULT_CONFIG,
): Promise<string | null> {
  const text = (event.text || '').trim();
  if (!text) return null;

  // Early exit: skip when graph is empty
  if (db.getNodeCount() === 0) {
    return null;
  }

  // Run hybrid search with input-specific timeout (HK-4)
  const results = await searchHybrid(
    db,
    text,
    config.inputSearchMaxResults,
    undefined,    // no category filter
    undefined,    // no subcategory filter
    config,
  );

  if (results.length === 0) return null;

  // Filter to high-scoring, high-trust results (HK-3)
  const highScore = results.filter(
    r => r.compositeScore >= config.inputSearchThreshold && r.node.trust === 'high',
  );

  if (highScore.length === 0) return null;

  // Prevent double injection (HK-7)
  const deduped = highScore.filter(r => !injectedNodeIds.has(r.node.id));
  if (deduped.length === 0) return null;

  // Track injected IDs
  deduped.forEach(r => injectedNodeIds.add(r.node.id));

  // Format as injection block
  return formatInputSurface(deduped);
}

/**
 * Clear the per-turn injection tracking.
 * Call at the start of each new agent turn.
 */
export function clearInjectedIds(): void {
  injectedNodeIds.clear();
}
