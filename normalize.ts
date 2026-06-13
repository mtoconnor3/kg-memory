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
// Edge type synonyms
// ---------------------------------------------------------------------------

const EDGE_TYPE_SYNONYMS: Record<string, string> = {
  // blocks
  'blocks': 'blocks',
  'blocking': 'blocks',
  'blocked-by': 'blocks',
  'blocks-on': 'blocks',
  // depends-on
  'dep-on': 'depends-on',
  'depends-on': 'depends-on',
  'dependent': 'depends-on',
  'dependency': 'depends-on',
  'dependencies': 'depends-on',
  'dependent-on': 'depends-on',
  // relates-to
  'relates-to': 'relates-to',
  'related': 'relates-to',
  'related-to': 'relates-to',
  'relates': 'relates-to',
  // contradicts
  'contradicts': 'contradicts',
  'contradiction': 'contradicts',
  'contradictory': 'contradicts',
  'contradicts-with': 'contradicts',
  // supersedes
  'supersedes': 'supersedes',
  'superseded': 'supersedes',
  'superseded-by': 'supersedes',
  'supersede': 'supersedes',
  // used-by
  'used-by': 'used-by',
  'uses': 'used-by',
  'used': 'used-by',
  'used-on': 'used-by',
  // implements
  'implements': 'implements',
  'implementation': 'implements',
  'implemented': 'implements',
  'implementing': 'implements',
  // causes
  'causes': 'causes',
  'caused-by': 'causes',
  'causality': 'causes',
  'causal': 'causes',
};

/**
 * Normalize an edge type string.
 * Steps: trim → lowercase → replace spaces/underscores with hyphens → synonym lookup.
 */
export function normalizeEdgeType(rawType: string): string {
  if (!rawType) return 'relates-to';
  const trimmed = rawType.trim().toLowerCase();
  if (!trimmed) return 'relates-to';
  const normalized = trimmed.replace(/[\s_]+/g, '-');
  return EDGE_TYPE_SYNONYMS[normalized] ?? normalized;
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
