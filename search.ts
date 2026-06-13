/**
 * Knowledge Graph — Search Layer (Phase 2: Vector Search)
 *
 * Handles LMStudio embedding API integration, hybrid search (FTS5 + vectors),
 * and ranking with configurable weights.
 */

import type { KnowledgeGraphDB, SearchHit } from './db.ts';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface SearchConfig {
  embeddingEndpoint: string;
  embeddingModel: string;
  maxResults: number;
  ftxF5Weight: number;
  vectorWeight: number;
  frequencyWeight: number;

  // Input hook config
  inputSearchThreshold: number;
  inputSearchMaxResults: number;
  inputSearchTimeout: number;
}

export const DEFAULT_CONFIG: SearchConfig = {
  embeddingEndpoint: 'http://192.168.1.1:1234/v1/embeddings',
  embeddingModel: 'nomic-embed-text-v1.5',
  maxResults: 10,
  ftxF5Weight: 0.4,
  vectorWeight: 0.5,
  frequencyWeight: 0.1,
  inputSearchThreshold: 0.45,
  inputSearchMaxResults: 3,
  inputSearchTimeout: 2000,
};

// ---------------------------------------------------------------------------
// Embedding API
// ---------------------------------------------------------------------------

/**
 * Call LMStudio to generate an embedding for text.
 * Returns the raw float array (768 dims for nomic-embed-text).
 */
export async function getEmbedding(
  db: KnowledgeGraphDB,
  text: string,
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<number[] | null> {
  try {
    const response = await fetch(config.embeddingEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.embeddingModel,
        input: text,
      }),
      signal: AbortSignal.timeout(10000), // 10s timeout
    });

    if (!response.ok) {
      console.warn(`[kg-memory] Embedding API returned ${response.status}: ${response.statusText}`);
      return null;
    }

    const data = await response.json() as { data: Array<{ embedding: number[] }> };
    return data.data?.[0]?.embedding ?? null;
  } catch (err) {
    console.warn('[kg-memory] Embedding API unavailable:', (err as Error).message);
    return null;
  }
}

/**
 * Store an embedding for a node (lazy: generated on demand).
 */
export async function storeEmbedding(
  db: KnowledgeGraphDB,
  nodeId: string,
  embedding: number[],
): Promise<void> {
  // Convert float array to binary blob for SQLite
  const buffer = Buffer.from(new Float32Array(embedding).buffer);

  try {
    db.db.prepare('INSERT OR REPLACE INTO node_vectors (node_id, embedding) VALUES (?, ?)').run(
      nodeId,
      buffer,
    );
  } catch (err) {
    console.warn('[kg-memory] Failed to store embedding:', (err as Error).message);
  }
}

/**
 * Retrieve a stored embedding for a node.
 */
export function getStoredEmbedding(db: KnowledgeGraphDB, nodeId: string): number[] | null {
  try {
    const row = db.db.prepare(
      'SELECT embedding FROM node_vectors WHERE node_id = ?',
    ).get(nodeId) as { embedding: Buffer } | undefined;

    if (!row) return null;

    // Convert BLOB back to float array
    const floatArray = new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4);
    return Array.from(floatArray);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Vector distance (cosine)
// ---------------------------------------------------------------------------

/**
 * Compute cosine distance between two vectors.
 * For nomic-embed-text (general embeddings with signed values),
 * the result is in [0, 2]:
 *   0 = identical (same direction)
 *   1 = orthogonal (unrelated)
 *   2 = exactly opposite
 */
export function cosineDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 1.0; // undefined direction

  // Cosine similarity: dot / (|a| * |b|), ranges [-1, 1]
  const cosineSim = dotProduct / magnitude;

  // Cosine distance: 1 - cosine_similarity, ranges [0, 2] for signed embeddings
  //   -1 (opposite) → distance 2
  //    0 (orthogonal) → distance 1
  //   +1 (identical) → distance 0
  return 1.0 - cosineSim;
}

/**
 * Normalize cosine distance to [0, 1] for combining with BM25.
 * Since lower distance = more similar, and BM25 is also lower = more similar,
 * we normalize by dividing by 2.0 (the max distance for signed embeddings).
 */
export function normalizeCosineDistance(distance: number): number {
  return Math.max(0, Math.min(1, distance / 2.0));
}

// ---------------------------------------------------------------------------
// Hybrid search
// ---------------------------------------------------------------------------

/**
 * Get all nodes from the database, optionally filtered by category/subcategory.
 * Used as fallback candidate set when FTS5 returns 0 results.
 */
export function getFilteredNodes(
  db: KnowledgeGraphDB,
  categories?: string[],
  subcategories?: string[],
): KnowledgeNode[] {
  return db.getAllNodes(categories, subcategories);
}

/**
 * Get the maximum frequency across all nodes (for normalization).
 */
export function getMaxFrequency(db: KnowledgeGraphDB): number {
  return db.getMaxFrequency();
}

/**
 * Hybrid search: FTS5 + vector cosine distance.
 *
 * When FTS5 returns results, scores them by BM25 + vector distance.
 * When FTS5 returns 0, scores ALL nodes by vector similarity (no early exit).
 * When LMStudio is unavailable, falls back to pure FTS5 (may return 0).
 */
export async function searchHybrid(
  db: KnowledgeGraphDB,
  query: string,
  maxResults: number = 10,
  categories?: string[],
  subcategories?: string[],
  config: SearchConfig = DEFAULT_CONFIG,
): Promise<SearchHit[]> {
  // Phase 0: Early exit — no nodes, no point calling LMStudio.
  // This prevents KV cache churn from embedding requests when the graph is empty.
  const nodeCount = db.getNodeCount();
  if (nodeCount === 0) {
    return [];
  }

  // Phase 1: Get FTS5 candidate set (may be empty)
  const ftsResults = db.searchFTS5(query, maxResults * 3, categories, subcategories);

  // Phase 2: Always try vector search (even if FTS5 returned 0)
  const queryEmbedding = await getEmbedding(db, query, config);

  if (!queryEmbedding) {
    // LMStudio unavailable — return pure FTS5 (may be 0)
    console.warn('[kg-memory] Embedding unavailable, falling back to pure FTS5');
    return ftsResults.slice(0, maxResults);
  }

  // Candidate set: use FTS5 results if available, otherwise ALL nodes
  // This ensures we always have nodes to score by vector similarity
  const candidateNodes = getFilteredNodes(db, categories, subcategories);
  const candidates = ftsResults.length > 0
    ? ftsResults
    : candidateNodes.map(node => ({
        node,
        bm25Score: 0,
        vectorScore: null,
        frequencyBoost: 0,
        freshnessScore: 0,
        compositeScore: 0,
        edges: db.getNodeEdges(node.id),
      }));

  // Phase 3: Score each candidate by vector distance
  const scoredResults: SearchHit[] = [];
  const maxFreq = getMaxFrequency(db);
  const now = Date.now();

  for (const candidate of candidates) {
    const { node } = candidate;

    // Get or generate the node's embedding
    let nodeEmbedding = getStoredEmbedding(db, node.id);

    if (!nodeEmbedding) {
      // Lazy-embed: generate and store embedding for this node
      nodeEmbedding = await getEmbedding(db, node.content, config);
      if (nodeEmbedding) {
        await storeEmbedding(db, node.id, nodeEmbedding);
      }
    }

    if (!nodeEmbedding) {
      // Could not generate embedding — skip this node
      continue;
    }

    const distance = cosineDistance(queryEmbedding, nodeEmbedding);
    // Invert: distance → similarity (higher = more relevant)
    const normalizedDistance = normalizeCosineDistance(distance);
    const vectorScore = 1.0 - normalizedDistance;

    // Compute BM25 score (0 if no FTS match)
    const bm25Score = (candidate as SearchHit).bm25Score ?? 0;

    // Compute frequency boost
    const frequencyBoost = maxFreq > 1
      ? Math.min(Math.log(node.frequency + 1) / Math.log(maxFreq + 1), 1.0)
      : 0;

    // Compute freshness
    const hoursSinceAccessed = (now - node.lastAccessedAt) / (1000 * 60 * 60);
    const freshness = 1.0 / (1.0 + hoursSinceAccessed / 720);

    // Composite score: BM25 + vector + frequency
    const compositeScore =
      (bm25Score * config.ftxF5Weight) +
      (vectorScore * config.vectorWeight) +
      (frequencyBoost * config.frequencyWeight);

    scoredResults.push({
      node,
      bm25Score,
      vectorScore,
      frequencyBoost,
      freshnessScore: freshness,
      compositeScore,
      edges: db.getNodeEdges(node.id),
    });
  }

  return scoredResults
    .sort((a, b) => b.compositeScore - a.compositeScore)
    .slice(0, maxResults);
}

// ---------------------------------------------------------------------------
// Ranking helpers
// ---------------------------------------------------------------------------

/**
 * Log-normalized frequency boost.
 * Prevents high-frequency nodes from dominating.
 */
export function frequencyBoost(frequency: number, maxFrequency: number): number {
  if (maxFrequency <= 1) return 0;
  return Math.min(Math.log(frequency + 1) / Math.log(maxFrequency + 1), 1.0);
}

/**
 * Exponential decay freshness score.
 * 30-day half-life: nodes not accessed in 30 days get half score.
 */
export function freshnessScore(lastAccessedAt: number, now: number): number {
  const hoursSinceAccessed = (now - lastAccessedAt) / (1000 * 60 * 60);
  return 1.0 / (1.0 + hoursSinceAccessed / 720); // 30 days = 720 hours
}

/**
 * Validate search configuration weights sum to ~1.0.
 */
export function validateConfig(config: Partial<SearchConfig>): SearchConfig {
  const merged = { ...DEFAULT_CONFIG, ...config };

  const total = merged.ftxF5Weight + merged.vectorWeight + merged.frequencyWeight;
  if (Math.abs(total - 1.0) > 0.01) {
    console.warn(`[kg-memory] Search weights sum to ${total.toFixed(3)}, expected 1.0. Using defaults.`);
    return { ...DEFAULT_CONFIG };
  }

  // Validate endpoint is a URL
  try {
    new URL(merged.embeddingEndpoint);
  } catch {
    console.warn(`[kg-memory] Invalid embeddingEndpoint: ${merged.embeddingEndpoint}. Using default.`);
    merged.embeddingEndpoint = DEFAULT_CONFIG.embeddingEndpoint;
  }

  // Validate inputSearchThreshold (0–1)
  if (typeof merged.inputSearchThreshold !== 'number' || merged.inputSearchThreshold < 0 || merged.inputSearchThreshold > 1) {
    console.warn(`[kg-memory] Invalid inputSearchThreshold. Using default 0.65.`);
    merged.inputSearchThreshold = DEFAULT_CONFIG.inputSearchThreshold;
  }

  // Validate inputSearchMaxResults (positive integer)
  if (!Number.isInteger(merged.inputSearchMaxResults) || merged.inputSearchMaxResults < 1) {
    console.warn(`[kg-memory] Invalid inputSearchMaxResults. Using default 3.`);
    merged.inputSearchMaxResults = DEFAULT_CONFIG.inputSearchMaxResults;
  }

  // Validate inputSearchTimeout (positive number)
  if (typeof merged.inputSearchTimeout !== 'number' || merged.inputSearchTimeout <= 0) {
    console.warn(`[kg-memory] Invalid inputSearchTimeout. Using default 2000.`);
    merged.inputSearchTimeout = DEFAULT_CONFIG.inputSearchTimeout;
  }

  return merged;
}
