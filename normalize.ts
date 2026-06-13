/**
 * Subcategory and edge-type normalization with synonym maps.
 * Prevents fragmentation from formatting/grammatical variations.
 */

// ---------------------------------------------------------------------------
// Subcategory synonyms
// ---------------------------------------------------------------------------

const SUBCATEGORY_SYNONYMS: Record<string, string> = {
  // pull-request
  'pr': 'pull-request',
  'pull-request': 'pull-request',
  'pullreq': 'pull-request',
  'pullreqs': 'pull-request',
  'pull-req': 'pull-request',
  // developer
  'developer': 'developer',
  'dev': 'developer',
  // stakeholder
  'stakeholder': 'stakeholder',
  'stakeholders': 'stakeholder',
  // contractor
  'contractor': 'contractor',
  'contractors': 'contractor',
  // team
  'team': 'team',
  'teams': 'team',
  // bug
  'bug': 'bug',
  'bugs': 'bug',
  // workaround
  'workaround': 'workaround',
  'workarounds': 'workaround',
  // limitation
  'limitation': 'limitation',
  'limitations': 'limitation',
  // framework
  'framework': 'framework',
  'frameworks': 'framework',
  // library
  'library': 'library',
  'libraries': 'library',
  // cli-tool
  'cli-tool': 'cli-tool',
  // platform
  'platform': 'platform',
  'platforms': 'platform',
  // service
  'service': 'service',
  'services': 'service',
  // module
  'module': 'module',
  'modules': 'module',
  // file
  'file': 'file',
  'files': 'file',
  // endpoint
  'endpoint': 'endpoint',
  'endpoints': 'endpoint',
  // config
  'config': 'config',
  'configs': 'config',
  'configuration': 'config',
  // database
  'database': 'database',
  'databases': 'database',
  // api
  'api': 'api',
  'apis': 'api',
  // infrastructure
  'infrastructure': 'infrastructure',
  'infra': 'infrastructure',
  // deployment
  'deployment': 'deployment',
  'deployments': 'deployment',
  // onboarding
  'onboarding': 'onboarding',
  // review
  'review': 'review',
  'reviews': 'review',
  // knowledge subcategories
  'fact': 'fact',
  'facts': 'fact',
  'decision': 'decision',
  'decisions': 'decision',
  'pattern': 'pattern',
  'patterns': 'pattern',
  'warning': 'warning',
  'warnings': 'warning',
  'preference': 'preference',
  'preferences': 'preference',
  // process subcategories
  'workflow': 'workflow',
  'workflows': 'workflow',
  'procedure': 'procedure',
  'procedures': 'procedure',
  'ritual': 'ritual',
  'rituals': 'ritual',
};

/**
 * Normalize a subcategory string.
 * Steps: trim → lowercase → replace spaces/underscores with hyphens → synonym lookup.
 */
export function normalizeSubcategory(sub: string | undefined): string | undefined {
  if (!sub) return undefined;
  const trimmed = sub.trim().toLowerCase();
  if (!trimmed) return undefined;
  const normalized = trimmed.replace(/[\s_]+/g, '-');
  return SUBCATEGORY_SYNONYMS[normalized] ?? normalized;
}

// ---------------------------------------------------------------------------
// Edge type synonyms with direction semantics (NM-1)
// ---------------------------------------------------------------------------

interface EdgeSynonymEntry {
  canonical: string;
  invert: boolean;  // If true, swap source/target when normalizing
}

const EDGE_TYPE_SYNONYMS: Record<string, EdgeSynonymEntry> = {
  // blocks
  'blocks': { canonical: 'blocks', invert: false },
  'blocking': { canonical: 'blocks', invert: false },
  'blocked-by': { canonical: 'blocks', invert: true },    // A blocked-by B → B blocks A
  'blocks-on': { canonical: 'blocks', invert: false },
  // depends-on
  'dep-on': { canonical: 'depends-on', invert: false },
  'depends-on': { canonical: 'depends-on', invert: false },
  'dependent': { canonical: 'depends-on', invert: false },
  'dependency': { canonical: 'depends-on', invert: false },
  'dependencies': { canonical: 'depends-on', invert: false },
  'dependent-on': { canonical: 'depends-on', invert: false },
  // relates-to (symmetric, no inversion needed)
  'relates-to': { canonical: 'relates-to', invert: false },
  'related': { canonical: 'relates-to', invert: false },
  'related-to': { canonical: 'relates-to', invert: false },
  'relates': { canonical: 'relates-to', invert: false },
  // contradicts
  'contradicts': { canonical: 'contradicts', invert: false },
  'contradiction': { canonical: 'contradicts', invert: false },
  'contradictory': { canonical: 'contradicts', invert: false },
  'contradicts-with': { canonical: 'contradicts', invert: false },
  // supersedes
  'supersedes': { canonical: 'supersedes', invert: false },
  'superseded': { canonical: 'supersedes', invert: true },      // A superseded B → B supersedes A
  'superseded-by': { canonical: 'supersedes', invert: true },   // A superseded-by B → B supersedes A
  'supersede': { canonical: 'supersedes', invert: false },
  // used-by
  'used-by': { canonical: 'used-by', invert: false },
  'uses': { canonical: 'used-by', invert: true },               // A uses B → B used-by A
  'used': { canonical: 'used-by', invert: false },
  'used-on': { canonical: 'used-by', invert: false },
  // implements
  'implements': { canonical: 'implements', invert: false },
  'implementation': { canonical: 'implements', invert: false },
  'implemented': { canonical: 'implements', invert: false },
  'implementing': { canonical: 'implements', invert: false },
  // causes
  'causes': { canonical: 'causes', invert: false },
  'caused-by': { canonical: 'causes', invert: true },           // A caused-by B → B causes A
  'causality': { canonical: 'causes', invert: false },
  'causal': { canonical: 'causes', invert: false },
};

/**
 * Normalize an edge type string.
 * Returns the canonical type and whether endpoints should be swapped (NM-1).
 */
export function normalizeEdgeType(rawType: string): { canonical: string; invert: boolean } {
  if (!rawType) return { canonical: 'relates-to', invert: false };
  const trimmed = rawType.trim().toLowerCase();
  if (!trimmed) return { canonical: 'relates-to', invert: false };
  const normalized = trimmed.replace(/[\s_]+/g, '-');

  const entry = EDGE_TYPE_SYNONYMS[normalized];
  if (entry) {
    return entry;
  }

  // Unknown type — allow free-form but don't invert (NM-2)
  return { canonical: normalized, invert: false };
}

/**
 * Check if an edge type is canonical (NM-2).
 */
export function isCanonicalEdgeType(type: string): boolean {
  return CANONICAL_EDGE_TYPES.includes(type as EdgeType);
}

// ---------------------------------------------------------------------------
// Category validation
// ---------------------------------------------------------------------------

export const VALID_CATEGORIES = [
  'knowledge', 'project', 'people', 'system', 'tool', 'error', 'process',
] as const;

export type Category = (typeof VALID_CATEGORIES)[number];

export function isValidCategory(cat: string): cat is Category {
  return VALID_CATEGORIES.includes(cat as Category);
}

// ---------------------------------------------------------------------------
// Canonical edge types (for reference / autocomplete)
// ---------------------------------------------------------------------------

export const CANONICAL_EDGE_TYPES = [
  'blocks', 'depends-on', 'relates-to', 'contradicts',
  'supersedes', 'used-by', 'implements', 'causes',
] as const;

export type EdgeType = (typeof CANONICAL_EDGE_TYPES)[number];
