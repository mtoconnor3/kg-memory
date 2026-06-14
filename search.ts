/**
 * Knowledge Graph — Search Layer (Phase 2: Vector Search)
 *
 * Handles LMStudio embedding API integration, hybrid search (FTS5 + vectors),
 * and ranking with configurable weights.
 */

import type { KnowledgeGraphDB, SearchHit, KnowledgeNode } from './db.ts';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface KgConfig {
  // Database
  graphPath: string;

  // Embedding
  embeddingEndpoint: string;
  embeddingModel: string;

  // Search
  maxResults: number;

  // Ranking weights (must sum to ~1.0)
  ftsF5Weight: number;
  vectorWeight: number;
  frequencyWeight: number;
  freshnessWeight: number;

  // Input hook config
  inputSearchThreshold: number;
  inputSearchMaxResults: number;
  inputSearchTimeout: number;

  // Injection
  injectionBudget: number;

  // Query log
  queryLogLimit: number;

  // Staleness
  staleNodeDays: number;
}

export const DEFAULT_CONFIG: KgConfig = {
  graphPath: '',
  embeddingEndpoint: 'http://10.1.1.145:1234/v1/embeddings',
  embeddingModel: 'nomic-embed-text-v1.5',
  maxResults: 10,
  ftsF5Weight: 0.4,
  vectorWeight: 0.3,
  frequencyWeight: 0.15,
  freshnessWeight: 0.15,
  inputSearchThreshold: 0.45,
  inputSearchMaxResults: 3,
  inputSearchTimeout: 2000,
  injectionBudget: 2000,
  queryLogLimit: 1000,
  staleNodeDays: 90,
};

// ---------------------------------------------------------------------------
// Embedding API
// ---------------------------------------------------------------------------

/**
 * Call LMStudio to generate embeddings for text.
 * Supports batch input (array of strings) for efficiency (SR-1).
 * Applies nomic task prefixes (SR-4).
 */
export async function getEmbedding(
  text: string | string[],
  config: KgConfig = DEFAULT_CONFIG,
  timeoutMs: number = 10000,
): Promise<number[] | null> {
  try {
    // Apply nomic task prefix (SR-4)
    const input = Array.isArray(text) ? text : [text];
    const prefixed = input.map(t => `search_document: ${t}`);

    const response = await fetch(config.embeddingEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.embeddingModel,
        input: prefixed,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) {
      console.warn(`[kg-memory] Embedding API returned ${response.status}: ${response.statusText}`);
      return null;
    }

    const data = await response.json() as { data: Array<{ embedding: number[] }> };
    if (!Array.isArray(data.data) || data.data.length === 0) return null;

    // Return first embedding for single-text calls, or first for batch
    return data.data[0]?.embedding ?? null;
  } catch (err) {
    console.warn('[kg-memory] Embedding API unavailable:', (err as Error).message);
    return null;
  }
}

/**
 * Get embedding for a search query (with search_query: prefix).
 */
export async function getQueryEmbedding(
  query: string,
  config: KgConfig = DEFAULT_CONFIG,
  timeoutMs: number = 10000,
): Promise<number[] | null> {
  try {
    const response = await fetch(config.embeddingEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.embeddingModel,
        input: `search_query: ${query}`,
      }),
      signal: AbortSignal.timeout(timeoutMs),
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
 * Batch-embed multiple texts at once (SR-1).
 * Returns array of embeddings (null for any that failed).
 */
export async function batchEmbed(
  texts: string[],
  config: KgConfig = DEFAULT_CONFIG,
  timeoutMs: number = 10000,
): Promise<Array<number[] | null>> {
  if (texts.length === 0) return [];

  try {
    // Apply nomic task prefix (SR-4)
    const prefixed = texts.map(t => `search_document: ${t}`);

    const response = await fetch(config.embeddingEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: config.embeddingModel,
        input: prefixed,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) {
      console.warn(`[kg-memory] Batch embedding API returned ${response.status}`);
      return texts.map(() => null);
    }

    const data = await response.json() as { data: Array<{ embedding: number[] }> };
    if (!Array.isArray(data.data)) return texts.map(() => null);

    return data.data.map(d => d.embedding ?? null);
  } catch (err) {
    console.warn('[kg-memory] Batch embedding failed:', (err as Error).message);
    return texts.map(() => null);
  }
}

// ---------------------------------------------------------------------------
// Vector distance (cosine)
// ---------------------------------------------------------------------------

/**
 * Compute cosine similarity between two vectors (SR-7: raw similarity, not distance).
 * Returns value in [0, 1] where 1 = identical, 0 = orthogonal or worse.
 * Handles dimension mismatch gracefully (SR-2: fail soft, not throw).
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    // Dimension mismatch — return 0 (no similarity) instead of throwing (SR-2)
    return 0;
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
  if (magnitude === 0) return 0; // undefined direction → no similarity

  // Cosine similarity: dot / (|a| * |b|), ranges [-1, 1]
  const cosineSim = dotProduct / magnitude;

  // Clamp to [0, 1] — negative similarity treated as 0 (SR-7)
  return Math.max(0, Math.min(1, cosineSim));
}

// ---------------------------------------------------------------------------
// Unified scoring (SR-3)
// ---------------------------------------------------------------------------

/**
 * Compute composite score for a search hit using unified formula.
 * Weights: { ftsF5Weight, vectorWeight, frequencyWeight, freshnessWeight }
 * When vector is unavailable, its weight is redistributed to BM25.
 */
export function computeCompositeScore(
  bm25Score: number,
  vectorScore: number | null,
  frequencyBoost: number,
  freshnessScore: number,
  config: KgConfig,
): number {
  const w = config;
  if (vectorScore !== null) {
    // Full hybrid scoring — all four terms
    return (
      (bm25Score * w.ftsF5Weight) +
      (vectorScore * w.vectorWeight) +
      (frequencyBoost * w.frequencyWeight) +
      (freshnessScore * w.freshnessWeight)
    );
  } else {
    // No vector — redistribute vectorWeight to BM25 proportionally
    const noVectorTotal = w.ftsF5Weight + w.vectorWeight;
    const bm25Share = noVectorTotal > 0 ? w.ftsF5Weight / noVectorTotal : 0;
    return (
      (bm25Score * noVectorTotal * bm25Share / w.ftsF5Weight) +
      (frequencyBoost * w.frequencyWeight) +
      (freshnessScore * w.freshnessWeight)
    );
  }
}

/**
 * Log-normalized frequency boost (DB-5: uses access_count).
 */
function computeFrequencyBoost(
  frequency: number,
  accessCount: number,
  maxFreq: number,
  maxAccess: number,
): number {
  // Blend frequency (write count) and access_count (read count)
  const freqScore = maxFreq > 1
    ? Math.min(Math.log(frequency + 1) / Math.log(maxFreq + 1), 1.0)
    : 0;
  const accessScore = maxAccess > 1
    ? Math.min(Math.log(accessCount + 1) / Math.log(maxAccess + 1), 1.0)
    : 0;
  // Weight access slightly more — it reflects actual retrieval usefulness
  return 0.4 * freqScore + 0.6 * accessScore;
}

/**
 * Exponential decay freshness score (30-day half-life).
 */
function computeFreshnessScore(lastAccessedAt: number, now: number): number {
  const hoursSinceAccessed = (now - lastAccessedAt) / (1000 * 60 * 60);
  return 1.0 / (1.0 + hoursSinceAccessed / 720);
}

// ---------------------------------------------------------------------------
// Hybrid search
// ---------------------------------------------------------------------------

/**
 * Hybrid search: FTS5 (BM25) + vector KNN (cosine) unioned as first-class sources.
 *
 * - FTS5 provides lexical candidates via BM25
 * - vec0 KNN provides semantic candidates (no longer gated by FTS)
 * - Candidate sets are unioned by node ID, then scored with composite formula
 * - On embedding unavailability, falls back to FTS-only (graceful degradation)
 * - No node-embedding in the query path — vectors are populated at write/index time
 */
export async function searchHybrid(
  db: KnowledgeGraphDB,
  query: string,
  maxResults: number = 10,
  categories?: string[],
  subcategories?: string[],
  config: KgConfig = DEFAULT_CONFIG,
): Promise<SearchHit[]> {
  // Phase 0: Early exit — no nodes
  const nodeCount = db.getNodeCount();
  if (nodeCount === 0) {
    return [];
  }

  const trimmedQuery = (query || '').trim();
  const now = Date.now();
  const maxFreq = db.getMaxFrequency();
  const maxAccess = db.getMaxAccessCount();
  const k = maxResults * 3; // fetch extra candidates for post-filtering

  // Phase 1: FTS5 lexical candidates
  const ftsResults = db.searchFTS5(query, k, categories, subcategories);

  // Phase 2: Try vector search
  const vectorTimeout = trimmedQuery.length < 50 ? config.inputSearchTimeout : 10000;
  const queryEmbedding = await getQueryEmbedding(query, config, vectorTimeout);

  if (!queryEmbedding) {
    // LMStudio unavailable — return FTS5 results with unified scoring (SR-3).
    // NOTE: This fallback is acceptable only at the curated-node scale.
    // vec_nodes KNN is required before the chunk tier lands.
    console.warn('[kg-memory] Embedding unavailable, falling back to pure FTS5');
    return ftsResults
      .map(hit => ({
        ...hit,
        compositeScore: computeCompositeScore(
          hit.bm25Score,
          null,
          computeFrequencyBoost(hit.node.frequency, hit.node.accessCount, maxFreq, maxAccess),
          hit.freshnessScore,
          config,
        ),
      }))
      .sort((a, b) => b.compositeScore - a.compositeScore)
      .slice(0, maxResults);
  }

  // Phase 3: KNN semantic candidates (first-class, not gated by FTS)
  const knnResults = db.knnSearch(queryEmbedding, k);

  // Phase 4: Union FTS and KNN candidates by node ID
  // For each unique candidate, compute all score components
  const candidateMap = new Map<string, {
    node: KnowledgeNode;
    bm25Score: number;
    vectorScore: number | null;
    knnDistance: number | null;
  }>();

  // Index FTS results by node ID
  for (const hit of ftsResults) {
    candidateMap.set(hit.node.id, {
      node: hit.node,
      bm25Score: hit.bm25Score,
      vectorScore: null,
      knnDistance: null,
    });
  }

  // Index KNN results by node ID
  for (const knn of knnResults) {
    const existing = candidateMap.get(knn.nodeId);
    if (existing) {
      // Already in FTS set — add KNN distance
      existing.knnDistance = knn.distance;
    } else {
      // Pure KNN hit — hydrate the node
      const node = db.getNode(knn.nodeId);
      if (node) {
        candidateMap.set(knn.nodeId, {
          node,
          bm25Score: 0,
          vectorScore: null,
          knnDistance: knn.distance,
        });
      }
    }
  }

  // Phase 5: Apply category/subcategory filters to unioned candidates
  type CandidateEntry = { node: KnowledgeNode; bm25Score: number; vectorScore: number | null; knnDistance: number | null };
  const filterCandidates = (candidates: Map<string, CandidateEntry>) => {
    const filtered = new Map<string, CandidateEntry>();
    for (const [id, c] of candidates) {
      if (categories && categories.length > 0 && !categories.includes(c.node.category)) continue;
      if (subcategories && subcategories.length > 0 && c.node.subcategory && !subcategories.includes(c.node.subcategory)) continue;
      filtered.set(id, c);
    }
    return filtered;
  };

  const filteredCandidates = filterCandidates(candidateMap);

  // Phase 6: Score each candidate
  const scoredResults: SearchHit[] = [];
  for (const candidate of filteredCandidates.values()) {
    const { node } = candidate;

    // Vector score: prefer KNN distance (1 - distance) if available,
    // otherwise compute cosineSimilarity from stored vector
    let vectorScore: number | null = null;
    if (candidate.knnDistance !== null) {
      // KNN hit — similarity = 1 - cosine distance
      vectorScore = Math.max(0, 1 - candidate.knnDistance);
    } else {
      // FTS-only hit — compute cosine from stored vector
      const vectorData = db.getVector(node.id);
      if (vectorData) {
        if (vectorData.model === config.embeddingModel && vectorData.dim === queryEmbedding.length) {
          try {
            vectorScore = cosineSimilarity(queryEmbedding, vectorData.embedding);
          } catch {
            vectorScore = null;
          }
        }
      }
    }

    // Frequency + freshness (unchanged)
    const frequencyBoost = computeFrequencyBoost(node.frequency, node.accessCount, maxFreq, maxAccess);
    const freshnessScore = computeFreshnessScore(node.lastAccessedAt, now);

    const compositeScore = computeCompositeScore(
      candidate.bm25Score,
      vectorScore,
      frequencyBoost,
      freshnessScore,
      config,
    );

    scoredResults.push({
      node,
      bm25Score: candidate.bm25Score,
      vectorScore,
      frequencyBoost,
      freshnessScore,
      compositeScore,
      edges: db.getNodeEdges(node.id),
    });
  }

  return scoredResults
    .sort((a, b) => b.compositeScore - a.compositeScore)
    .slice(0, maxResults);
}

// ---------------------------------------------------------------------------
// Config validation (per-field, not all-or-nothing — CFG-2)
// ---------------------------------------------------------------------------

/**
 * Validate and merge config with defaults.
 * Each field is validated independently — a bad weight doesn't discard other values (CFG-2).
 */
export function validateConfig(config: Partial<KgConfig>): KgConfig {
  const merged = { ...DEFAULT_CONFIG, ...config };

  // Support deprecated ftxF5Weight key (CFG-1)
  if ('ftxF5Weight' in config && !('ftsF5Weight' in config)) {
    merged.ftsF5Weight = (config as any).ftxF5Weight;
    console.warn('[kg-memory] Deprecated key "ftxF5Weight" — use "ftsF5Weight" instead');
  }

  // Validate and clamp weights independently (CFG-2)
  if (typeof merged.ftsF5Weight !== 'number' || merged.ftsF5Weight < 0 || merged.ftsF5Weight > 1) {
    console.warn(`[kg-memory] Invalid ftsF5Weight. Using default ${DEFAULT_CONFIG.ftsF5Weight}.`);
    merged.ftsF5Weight = DEFAULT_CONFIG.ftsF5Weight;
  }
  if (typeof merged.vectorWeight !== 'number' || merged.vectorWeight < 0 || merged.vectorWeight > 1) {
    console.warn(`[kg-memory] Invalid vectorWeight. Using default ${DEFAULT_CONFIG.vectorWeight}.`);
    merged.vectorWeight = DEFAULT_CONFIG.vectorWeight;
  }
  if (typeof merged.frequencyWeight !== 'number' || merged.frequencyWeight < 0 || merged.frequencyWeight > 1) {
    console.warn(`[kg-memory] Invalid frequencyWeight. Using default ${DEFAULT_CONFIG.frequencyWeight}.`);
    merged.frequencyWeight = DEFAULT_CONFIG.frequencyWeight;
  }
  if (typeof merged.freshnessWeight !== 'number' || merged.freshnessWeight < 0 || merged.freshnessWeight > 1) {
    console.warn(`[kg-memory] Invalid freshnessWeight. Using default ${DEFAULT_CONFIG.freshnessWeight}.`);
    merged.freshnessWeight = DEFAULT_CONFIG.freshnessWeight;
  }

  // Warn if weights don't sum to ~1.0 (but don't discard other config)
  const total = merged.ftsF5Weight + merged.vectorWeight + merged.frequencyWeight + merged.freshnessWeight;
  if (Math.abs(total - 1.0) > 0.01) {
    console.warn(`[kg-memory] Search weights sum to ${total.toFixed(3)}, expected 1.0. Scores may be off.`);
  }

  // Validate endpoint
  try {
    new URL(merged.embeddingEndpoint);
  } catch {
    console.warn(`[kg-memory] Invalid embeddingEndpoint: ${merged.embeddingEndpoint}. Using default.`);
    merged.embeddingEndpoint = DEFAULT_CONFIG.embeddingEndpoint;
  }

  // Validate inputSearchThreshold (CFG-4: correct default)
  if (typeof merged.inputSearchThreshold !== 'number' || merged.inputSearchThreshold < 0 || merged.inputSearchThreshold > 1) {
    console.warn(`[kg-memory] Invalid inputSearchThreshold. Using default ${DEFAULT_CONFIG.inputSearchThreshold}.`);
    merged.inputSearchThreshold = DEFAULT_CONFIG.inputSearchThreshold;
  }

  // Validate inputSearchMaxResults
  if (!Number.isInteger(merged.inputSearchMaxResults) || merged.inputSearchMaxResults < 1) {
    console.warn(`[kg-memory] Invalid inputSearchMaxResults. Using default ${DEFAULT_CONFIG.inputSearchMaxResults}.`);
    merged.inputSearchMaxResults = DEFAULT_CONFIG.inputSearchMaxResults;
  }

  // Validate inputSearchTimeout (CFG-5: now threaded through)
  if (typeof merged.inputSearchTimeout !== 'number' || merged.inputSearchTimeout <= 0) {
    console.warn(`[kg-memory] Invalid inputSearchTimeout. Using default ${DEFAULT_CONFIG.inputSearchTimeout}.`);
    merged.inputSearchTimeout = DEFAULT_CONFIG.inputSearchTimeout;
  }

  // Validate injectionBudget (TS-3, CFG-2)
  if (typeof merged.injectionBudget !== 'number' || merged.injectionBudget <= 0) {
    console.warn(`[kg-memory] Invalid injectionBudget. Using default ${DEFAULT_CONFIG.injectionBudget}.`);
    merged.injectionBudget = DEFAULT_CONFIG.injectionBudget;
  }

  // Validate queryLogLimit
  if (typeof merged.queryLogLimit !== 'number' || merged.queryLogLimit < 10) {
    console.warn(`[kg-memory] Invalid queryLogLimit. Using default ${DEFAULT_CONFIG.queryLogLimit}.`);
    merged.queryLogLimit = DEFAULT_CONFIG.queryLogLimit;
  }

  // Validate staleNodeDays (CFG-6)
  if (typeof merged.staleNodeDays !== 'number' || merged.staleNodeDays < 1) {
    console.warn(`[kg-memory] Invalid staleNodeDays. Using default ${DEFAULT_CONFIG.staleNodeDays}.`);
    merged.staleNodeDays = DEFAULT_CONFIG.staleNodeDays;
  }

  return merged;
}
