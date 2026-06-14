/**
 * Knowledge Graph Memory Extension — Entry Point
 *
 * Registers tools, hooks, and commands with Pi's extension system:
 *
 * Tools:  kg_add, kg_search, kg_link, kg_neighbors, kg_get, kg_delete, kg_query
 * Hooks:  session_start, before_agent_start, context, session_before_compact, session_shutdown
 * Commands: /kg (graph overview), /kg-query (analytics)
 */

import path from 'path';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { openKnowledgeGraph } from './db.ts';
import {
  kgAdd,
  kgSearch,
  kgLink,
  kgNeighbors,
  kgGet,
  kgDelete,
  kgQuery,
} from './tools.ts';
import {
  onSessionStart,
  onBeforeAgentStart,
  onContext,
  onSessionBeforeCompact,
  onSessionShutdown,
  onInput,
  clearInjectedIds,
} from './hooks.ts';
import {
  generateGraphOverview,
  generateFormattedReport,
  logQuery,
  pruneOldEntries,
} from './logging.ts';
import {
  DEFAULT_CONFIG,
  validateConfig,
  batchEmbed,
  type KgConfig,
} from './search.ts';
import { normalizeSubcategory, normalizeEdgeType } from './normalize.ts';

// ---------------------------------------------------------------------------
// Vector backfill (B.4: one-time migration for existing nodes)
// ---------------------------------------------------------------------------

/**
 * Embed nodes that lack vectors, batched to avoid blocking startup.
 * Idempotent — only processes nodes with no existing vector.
 */
async function backfillMissingVectors(db: ReturnType<typeof openKnowledgeGraph>, config: KgConfig): Promise<void> {
  // Find nodes without vectors
  const rows = db.getDb().prepare(`
    SELECT n.id, n.content FROM nodes n
    LEFT JOIN node_vectors nv ON nv.node_id = n.id
    WHERE nv.node_id IS NULL
  `).all() as Array<{ id: string; content: string }>;

  if (rows.length === 0) {
    return; // Nothing to backfill
  }

  console.log(`[kg-memory] Backfilling vectors for ${rows.length} node(s) without embeddings...`);

  // Batch embed and store
  const batchSize = 25;
  for (let i = 0; i < rows.length; i += batchSize) {
    const batch = rows.slice(i, i + batchSize);
    const contents = batch.map(r => r.content);
    const embeddings = await batchEmbed(contents, config, 10000);

    for (let j = 0; j < batch.length; j++) {
      const embedding = embeddings[j];
      if (embedding) {
        db.storeVector(batch[j].id, embedding, config.embeddingModel);
      }
    }

    // Yield between batches to avoid blocking
    await new Promise(resolve => setImmediate(resolve));
  }

  console.log(`[kg-memory] Vector backfill complete for ${rows.length} node(s)`);
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

function resolveGraphPath(): string {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '.';
  return path.join(homeDir, '.pi', 'agent', 'memory', 'kg.db');
}

async function loadConfig(): Promise<Record<string, any>> {
  try {
    const settingsPath = path.join(process.env.HOME || '.', '.pi', 'agent', 'settings.json');
    const fs = await import('fs');
    const raw = fs.readFileSync(settingsPath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Extension initialization
// ---------------------------------------------------------------------------

export default async function (pi: ExtensionAPI): Promise<void> {
  const config = await loadConfig();
  const kgConfig = config.kgMemory || {};

  // Resolve path: config overrides default
  const graphPath = kgConfig.graphPath || resolveGraphPath();

  // Validate and merge config (per-field validation — CFG-2)
  const searchConfig = validateConfig(kgConfig);

  // Open (or create) the database
  const db = openKnowledgeGraph(graphPath);

  console.log(`[kg-memory] Knowledge graph initialized: ${graphPath}`);
  console.log(`[kg-memory] Embedding: ${searchConfig.embeddingEndpoint} (model: ${searchConfig.embeddingModel})`);

  // Backfill vec_nodes KNN index from existing node_vectors (idempotent)
  const backfilled = db.backfillVecIndex();
  if (backfilled > 0) {
    console.log(`[kg-memory] Backfilled ${backfilled} vectors into vec_nodes KNN index`);
  }

  // Embed any nodes that lack vectors (one-time backfill for existing content)
  await backfillMissingVectors(db, searchConfig);

  // Register tools
  registerTools(db, searchConfig, pi);

  // Register hooks
  registerHooks(db, searchConfig, pi);

  // Register commands
  registerCommands(db, searchConfig, pi);

  // Auto-prune query log on startup
  pruneOldEntries(db, searchConfig.queryLogLimit);
}

// ---------------------------------------------------------------------------
// Tool registration
// ---------------------------------------------------------------------------

function registerTools(db: ReturnType<typeof openKnowledgeGraph>, config: KgConfig, pi: ExtensionAPI): void {
  // kg_add
  pi.registerTool({
    name: 'kg_add',
    description: 'Add a node to the knowledge graph. Normalizes category, subcategory, and edge types. Deduplicates by (category, subcategory, content) identity.',
    parameters: {
      type: 'object',
      properties: {
        category: {
          type: 'string',
          description: 'Node category: knowledge, project, people, system, tool, error, process',
        },
        content: {
          type: 'string',
          description: 'The knowledge content (fact, decision, warning, etc.). Max 4000 characters.',
        },
        subcategory: {
          type: 'string',
          description: 'Optional subcategory (e.g., fact, bug, developer). Normalized via synonym map.',
        },
        properties: {
          type: 'object',
          description: 'Optional freeform metadata as key-value pairs.',
        },
      },
      required: ['category', 'content'],
    },
    execute: async (_toolCallId: string, params: Record<string, any>) => {
      const result = kgAdd(db, {
        category: params.category,
        content: params.content,
        subcategory: params.subcategory,
        properties: params.properties,
      });
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    },
  });

  // kg_search
  pi.registerTool({
    name: 'kg_search',
    description: 'Search the knowledge graph. Hybrid search: FTS5 (BM25) + vector (if LMStudio available). Falls back to pure FTS5.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query text.',
        },
        maxResults: {
          type: 'number',
          description: 'Maximum number of results (default: 10).',
        },
        categories: {
          type: 'array',
          items: { type: 'string' },
          description: 'Filter by categories (e.g., ["knowledge", "error"]).',
        },
        subcategories: {
          type: 'array',
          items: { type: 'string' },
          description: 'Filter by subcategories (e.g., ["bug", "fact"]).',
        },
      },
    },
    execute: async (_toolCallId: string, params: Record<string, any>) => {
      const result = await kgSearch(db, {
        query: params.query ?? '',
        maxResults: params.maxResults,
        categories: params.categories,
        subcategories: params.subcategories,
      }, config);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    },
  });

  // kg_link
  pi.registerTool({
    name: 'kg_link',
    description: 'Create an edge between two nodes. Normalizes edge types (blocks, depends-on, relates-to, etc.). Inverse types (blocked-by, caused-by) swap endpoints.',
    parameters: {
      type: 'object',
      properties: {
        sourceId: {
          type: 'string',
          description: 'ID of the source node.',
        },
        targetId: {
          type: 'string',
          description: 'ID of the target node.',
        },
        type: {
          type: 'string',
          description: 'Edge type (e.g., "blocks", "depends-on", "relates-to"). Normalized via synonym map.',
        },
      },
      required: ['sourceId', 'targetId', 'type'],
    },
    execute: async (_toolCallId: string, params: Record<string, any>) => {
      const result = kgLink(db, {
        sourceId: params.sourceId,
        targetId: params.targetId,
        type: params.type,
      });
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    },
  });

  // kg_neighbors
  pi.registerTool({
    name: 'kg_neighbors',
    description: 'Find connected nodes via BFS traversal through edges.',
    parameters: {
      type: 'object',
      properties: {
        nodeId: {
          type: 'string',
          description: 'ID of the node to find neighbors for.',
        },
        maxDepth: {
          type: 'number',
          description: 'Maximum BFS depth (default: 2).',
        },
      },
      required: ['nodeId'],
    },
    execute: async (_toolCallId: string, params: Record<string, any>) => {
      const result = kgNeighbors(db, params.nodeId, params.maxDepth);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    },
  });

  // kg_get
  pi.registerTool({
    name: 'kg_get',
    description: 'Get full node details including all incident edges.',
    parameters: {
      type: 'object',
      properties: {
        nodeId: {
          type: 'string',
          description: 'ID of the node to retrieve.',
        },
      },
      required: ['nodeId'],
    },
    execute: async (_toolCallId: string, params: Record<string, any>) => {
      const result = kgGet(db, params.nodeId);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    },
  });

  // kg_delete
  pi.registerTool({
    name: 'kg_delete',
    description: 'Remove a node and all incident edges from the knowledge graph.',
    parameters: {
      type: 'object',
      properties: {
        nodeId: {
          type: 'string',
          description: 'ID of the node to delete.',
        },
      },
      required: ['nodeId'],
    },
    execute: async (_toolCallId: string, params: Record<string, any>) => {
      const result = kgDelete(db, params.nodeId);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    },
  });

  // kg_query
  pi.registerTool({
    name: 'kg_query',
    description: 'Query the query log for analytics: graph stats, most surfaced nodes, gaps, distribution.',
    parameters: {
      type: 'object',
      properties: {
        queryType: {
          type: 'string',
          description: 'Filter by query type (search, kg_add, kg_link, etc.).',
        },
        maxResults: {
          type: 'number',
          description: 'Maximum number of results (default: 20).',
        },
      },
      required: [],
    },
    execute: async (_toolCallId: string, params: Record<string, any>) => {
      const result = kgQuery(db, {
        queryType: params.queryType,
        maxResults: params.maxResults,
      });
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    },
  });
}

// ---------------------------------------------------------------------------
// Hook registration
// ---------------------------------------------------------------------------

function registerHooks(db: ReturnType<typeof openKnowledgeGraph>, config: KgConfig, pi: ExtensionAPI): void {
  // session_start
  pi.on('session_start', (context: Record<string, any>) => {
    const result = onSessionStart(db, {
      sessionId: context.sessionId || 'unknown',
      project: context.project,
    }, config);

    if (result.notification) {
      pi.ui?.notify?.(result.notification);
    }
  });

  // before_agent_start
  pi.on('before_agent_start', async (context: Record<string, any>) => {
    // Clear per-turn injection tracking (HK-7)
    clearInjectedIds();

    const result = await onBeforeAgentStart(db, {
      sessionId: context.sessionId || 'unknown',
      previousSessionId: context.previousSessionId,
    }, config);

    if (result.injection) {
      // Inject into system prompt (HK-3: provenance-gated)
      pi.systemPrompt?.inject?.(result.injection);
    }
  });

  // context (after compaction)
  pi.on('context', async (context: Record<string, any>) => {
    const result = await onContext(db, {
      sessionId: context.sessionId || 'unknown',
      compactedContent: context.compactedContent || '',
    }, config);

    if (result.injection) {
      pi.systemPrompt?.inject?.(result.injection);
    }
  });

  // session_before_compact
  pi.on('session_before_compact', (context: Record<string, any>) => {
    const result = onSessionBeforeCompact(db, {
      sessionId: context.sessionId || 'unknown',
      sessionHistory: context.sessionHistory || [],
    });

    if (result.injection) {
      pi.systemPrompt?.inject?.(result.injection);
    }
  });

  // session_shutdown
  pi.on('session_shutdown', (context: Record<string, any>) => {
    onSessionShutdown(db, {
      sessionId: context.sessionId || 'unknown',
    }, config);
  });

  // input — Auto-surface relevant KG nodes on user input
  pi.on('input', async (event: Record<string, any>) => {
    try {
      const injection = await onInput(db, {
        text: event.text ?? '',
        images: event.images,
        source: event.source,
        streamingBehavior: event.streamingBehavior,
      }, config);

      if (injection) {
        pi.systemPrompt?.inject?.(injection);
      }
    } catch (err) {
      // Fail silently — don't break the agent turn
      console.warn('[kg-memory] Input hook failed:', (err as Error).message);
    }
  });
}

// ---------------------------------------------------------------------------
// Command registration (LG-4: differentiated commands)
// ---------------------------------------------------------------------------

function registerCommands(db: ReturnType<typeof openKnowledgeGraph>, config: KgConfig, pi: ExtensionAPI): void {
  // /kg — Graph overview (node count, edges, categories, top nodes)
  pi.registerCommand('kg', {
    description: 'Show knowledge graph overview: node count, edge count, category distribution, most surfaced nodes.',
    handler: async () => {
      const report = generateGraphOverview(db);
      return report;
    },
  });

  // /kg-query — Query log analytics (gaps, query types, agent actions, growth)
  pi.registerCommand('kg-query', {
    description: 'Show query log analytics: most surfaced nodes, zero-result gaps, category distribution, query types, agent actions.',
    handler: async () => {
      const report = generateFormattedReport(db, '', { queryLogLimit: config.queryLogLimit });
      return report;
    },
  });
}

// ---------------------------------------------------------------------------
// Exports (for testing)
// ---------------------------------------------------------------------------

export { openKnowledgeGraph, kgAdd, kgSearch, kgLink, kgNeighbors, kgGet, kgDelete, kgQuery };
export { onSessionStart, onBeforeAgentStart, onContext, onSessionBeforeCompact, onSessionShutdown, onInput, clearInjectedIds };
export { generateGraphOverview, generateFormattedReport };
export { normalizeSubcategory, normalizeEdgeType };
export { DEFAULT_CONFIG, validateConfig };
