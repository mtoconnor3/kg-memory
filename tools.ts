/**
 * Knowledge Graph — Tool Registrations
 *
 * Exposes KG operations to the LLM via Pi's tool system:
 *   kg_add, kg_search, kg_link, kg_neighbors, kg_delete, kg_get, kg_query
 */

import { createHash } from 'crypto';
import type { KnowledgeGraphDB, KnowledgeNode, KnowledgeEdge } from './db.ts';
import { searchHybrid } from './search.ts';
import {
  normalizeSubcategory,
  normalizeEdgeType,
  isValidCategory,
  VALID_CATEGORIES,
  CANONICAL_EDGE_TYPES,
} from './normalize.ts';

// ---------------------------------------------------------------------------
// Tool parameters / return types
// ---------------------------------------------------------------------------

export interface AddNodeParams {
  category: string;
  content: string;
  subcategory?: string;
  properties?: Record<string, string>;
}

export interface AddNodeResult {
  success: boolean;
  nodeId?: string;
  created: boolean;
  message: string;
}

export interface SearchNodeParams {
  query: string;
  maxResults?: number;
  categories?: string[];
  subcategories?: string[];
}

export interface SearchNodeResult {
  success: boolean;
  results: Array<{
    nodeId: string;
    category: string;
    subcategory: string | null;
    content: string;
    score: number;
    edges: Array<{ sourceId: string; targetId: string; type: string }>;
  }>;
  message: string;
}

export interface LinkNodeParams {
  sourceId: string;
  targetId: string;
  type: string;
}

export interface LinkNodeResult {
  success: boolean;
  created: boolean;
  message: string;
}

export interface NeighborsResult {
  success: boolean;
  neighbors: Array<{ nodeId: string; edges: Array<{ sourceId: string; targetId: string; type: string }> }>;
  message: string;
}

export interface GetNodeResult {
  success: boolean;
  node?: KnowledgeNode;
  edges?: KnowledgeEdge[];
  message: string;
}

export interface DeleteNodeResult {
  success: boolean;
  deleted: boolean;
  message: string;
}

export interface QueryLogParams {
  queryType?: string;
  maxResults?: number;
}

export interface QueryLogResult {
  success: boolean;
  stats: Record<string, any>;
  mostSurfaced: Array<{ nodeId: string; hits: number }>;
  gaps: Array<{ query: string; count: number }>;
  distribution: Array<{ category: string; subcategory: string; count: number }>;
  message: string;
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

/**
 * kg_add — Add a node to the knowledge graph.
 *
 * Normalizes category (validates), subcategory (synonym map), and content.
 * Deduplicates by content_hash — updates existing node instead of creating duplicate.
 */
export function kgAdd(
  db: KnowledgeGraphDB,
  params: AddNodeParams,
): AddNodeResult {
  // Validate category
  if (!isValidCategory(params.category)) {
    return {
      success: false,
      created: false,
      message: `Invalid category "${params.category}". Valid categories: ${VALID_CATEGORIES.join(', ')}`,
    };
  }

  // Normalize subcategory
  const normalizedSubcategory = normalizeSubcategory(params.subcategory);

  // Save node (dedup by content_hash)
  const result = db.saveNode({
    category: params.category,
    subcategory: normalizedSubcategory,
    content: params.content,
    properties: params.properties,
    createdAt: Date.now(),
    lastAccessedAt: Date.now(),
    frequency: 0,
    contentHash: createHash('sha256').update(params.content).digest('hex'),
  });

  // Log the operation
  db.logQuery({
    query: params.content,
    queryType: 'kg_add',
    resultsReturned: 1,
    injectedIds: [result.id],
    injectedTokenBudget: params.content.length,
  });

  if (result.created) {
    return {
      success: true,
      nodeId: result.id,
      created: true,
      message: `Node created: ${result.id} [${params.category}${normalizedSubcategory ? '/' + normalizedSubcategory : ''}]`,
    };
  } else {
    return {
      success: true,
      nodeId: result.id,
      created: false,
      message: `Node updated (dedup): ${result.id} [${params.category}${normalizedSubcategory ? '/' + normalizedSubcategory : ''}] — frequency increased`,
    };
  }
}

/**
 * kg_search — Search the knowledge graph.
 *
 * Hybrid search: FTS5 (BM25) + vector (if LMStudio available).
 * Falls back to pure FTS5 if embedding API is unavailable.
 * Supports filtering by categories and subcategories.
 */
export async function kgSearch(
  db: KnowledgeGraphDB,
  params: SearchNodeParams,
): Promise<SearchNodeResult> {
  const maxResults = params.maxResults || 10;

  const results = await searchHybrid(db, params.query, maxResults, params.categories, params.subcategories);

  // Log the search
  db.logQuery({
    query: params.query,
    queryType: 'search',
    resultsReturned: results.length,
    relevanceScore: results.length > 0 ? results[0].compositeScore : undefined,
    injectedIds: results.map(r => r.node.id),
    injectedTokenBudget: results.reduce((sum, r) => sum + r.node.content.length, 0),
  });

  return {
    success: true,
    results: results.map(r => ({
      nodeId: r.node.id,
      category: r.node.category,
      subcategory: r.node.subcategory,
      content: r.node.content,
      score: r.compositeScore,
      edges: r.edges.map(e => ({ sourceId: e.sourceId, targetId: e.targetId, type: e.type })),
    })),
    message: results.length > 0
      ? `Found ${results.length} result(s). Top score: ${results[0].compositeScore.toFixed(3)}`
      : `No results found for "${params.query}".`,
  };
}

/**
 * kg_link — Create an edge between two nodes.
 *
 * Normalizes edge type (synonym map) to prevent fragmentation.
 */
export function kgLink(
  db: KnowledgeGraphDB,
  params: LinkNodeParams,
): LinkNodeResult {
  // Validate nodes exist
  const source = db.getNode(params.sourceId);
  if (!source) {
    return { success: false, created: false, message: `Source node not found: ${params.sourceId}` };
  }

  const target = db.getNode(params.targetId);
  if (!target) {
    return { success: false, created: false, message: `Target node not found: ${params.targetId}` };
  }

  // Normalize edge type
  const normalizedType = normalizeEdgeType(params.type);

  // Save edge (dedup by source + target + type)
  const result = db.saveEdge({
    sourceId: params.sourceId,
    targetId: params.targetId,
    type: normalizedType,
    frequency: 0,
  });

  // Log the operation
  db.logQuery({
    query: `${params.sourceId} → ${params.targetId} (${normalizedType})`,
    queryType: 'kg_link',
    resultsReturned: 1,
    injectedIds: [params.sourceId, params.targetId],
  });

  return {
    success: true,
    created: result.created,
    message: result.created
      ? `Edge created: ${params.sourceId} → ${params.targetId} [${normalizedType}]`
      : `Edge updated (dedup): ${params.sourceId} → ${params.targetId} [${normalizedType}] — frequency increased`,
  };
}

/**
 * kg_neighbors — Find connected nodes via BFS traversal.
 */
export function kgNeighbors(
  db: KnowledgeGraphDB,
  nodeId: string,
  maxDepth: number = 2,
): NeighborsResult {
  const node = db.getNode(nodeId);
  if (!node) {
    return { success: false, neighbors: [], message: `Node not found: ${nodeId}` };
  }

  const neighbors = db.getNeighbors(nodeId, maxDepth);

  // Log the operation
  db.logQuery({
    query: `neighbors(${nodeId}, depth=${maxDepth})`,
    queryType: 'kg_neighbors',
    resultsReturned: neighbors.length,
    injectedIds: neighbors.map(n => n.nodeId),
  });

  return {
    success: true,
    neighbors: neighbors.map(n => ({
      nodeId: n.nodeId,
      edges: n.edges.map(e => ({ sourceId: e.sourceId, targetId: e.targetId, type: e.type })),
    })),
    message: `Found ${neighbors.length} neighbor(s) within depth ${maxDepth}`,
  };
}

/**
 * kg_get — Get full node details including edges.
 */
export function kgGet(
  db: KnowledgeGraphDB,
  nodeId: string,
): GetNodeResult {
  const node = db.getNode(nodeId);
  if (!node) {
    return { success: false, message: `Node not found: ${nodeId}` };
  }

  const edges = db.getNodeEdges(nodeId);

  // Log the operation
  db.logQuery({
    query: `get(${nodeId})`,
    queryType: 'kg_get',
    resultsReturned: 1,
    injectedIds: [nodeId],
  });

  return {
    success: true,
    node,
    edges,
    message: `Node retrieved: ${nodeId} [${node.category}${node.subcategory ? '/' + node.subcategory : ''}]`,
  };
}

/**
 * kg_delete — Remove a node and all incident edges.
 */
export function kgDelete(
  db: KnowledgeGraphDB,
  nodeId: string,
): DeleteNodeResult {
  const node = db.getNode(nodeId);
  if (!node) {
    return { success: false, deleted: false, message: `Node not found: ${nodeId}` };
  }

  const deleted = db.deleteNode(nodeId);

  // Log the operation
  db.logQuery({
    query: `delete(${nodeId})`,
    queryType: 'kg_delete',
    resultsReturned: deleted ? 1 : 0,
    injectedIds: deleted ? [nodeId] : undefined,
  });

  return {
    success: true,
    deleted,
    message: deleted
      ? `Node deleted: ${nodeId}`
      : `Node not found: ${nodeId}`,
  };
}

/**
 * kg_query — Query the query log for analytics.
 */
export function kgQuery(
  db: KnowledgeGraphDB,
  params: QueryLogParams,
): QueryLogResult {
  const stats = db.getGraphStats();
  const mostSurfaced = db.getMostSurfacedNodes(params.maxResults || 20);
  const gaps = db.getZeroResultQueries(params.maxResults || 20);
  const distribution = db.getCategoryDistribution();

  return {
    success: true,
    stats,
    mostSurfaced,
    gaps,
    distribution,
    message: `Graph: ${stats.nodeCount} nodes, ${stats.edgeCount} edges, ${stats.queryLogSize} log entries`,
  };
}
