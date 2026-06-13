import { describe, it, expect } from 'vitest';
import {
  normalizeSubcategory,
  normalizeEdgeType,
  isValidCategory,
  VALID_CATEGORIES,
  CANONICAL_EDGE_TYPES,
  isCanonicalEdgeType,
} from '../normalize.ts';

describe('normalizeSubcategory', () => {
  it('handles undefined', () => {
    expect(normalizeSubcategory(undefined)).toBeUndefined();
  });

  it('handles empty string', () => {
    expect(normalizeSubcategory('')).toBeUndefined();
  });

  it('handles whitespace-only', () => {
    expect(normalizeSubcategory('   ')).toBeUndefined();
  });

  it('lowercases and replaces spaces with hyphens', () => {
    expect(normalizeSubcategory('Pull Request')).toBe('pull-request');
  });

  it('replaces underscores with hyphens', () => {
    expect(normalizeSubcategory('pull_request')).toBe('pull-request');
  });

  it('applies synonym map', () => {
    expect(normalizeSubcategory('pr')).toBe('pull-request');
    expect(normalizeSubcategory('dev')).toBe('developer');
    expect(normalizeSubcategory('bugs')).toBe('bug');
    expect(normalizeSubcategory('infra')).toBe('infrastructure');
  });

  it('passes through unknown subcategories', () => {
    expect(normalizeSubcategory('custom-thing')).toBe('custom-thing');
  });
});

describe('normalizeEdgeType', () => {
  it('handles undefined', () => {
    const result = normalizeEdgeType('');
    expect(result.canonical).toBe('relates-to');
    expect(result.invert).toBe(false);
  });

  it('handles empty string', () => {
    const result = normalizeEdgeType('');
    expect(result.canonical).toBe('relates-to');
    expect(result.invert).toBe(false);
  });

  it('normalizes "blocks" synonyms', () => {
    expect(normalizeEdgeType('blocks').canonical).toBe('blocks');
    expect(normalizeEdgeType('blocking').canonical).toBe('blocks');
    expect(normalizeEdgeType('blocks-on').canonical).toBe('blocks');
  });

  it('normalizes "depends-on" synonyms', () => {
    expect(normalizeEdgeType('depends-on').canonical).toBe('depends-on');
    expect(normalizeEdgeType('dep-on').canonical).toBe('depends-on');
    expect(normalizeEdgeType('dependency').canonical).toBe('depends-on');
  });

  it('normalizes "relates-to" synonyms', () => {
    expect(normalizeEdgeType('relates-to').canonical).toBe('relates-to');
    expect(normalizeEdgeType('related').canonical).toBe('relates-to');
    expect(normalizeEdgeType('related-to').canonical).toBe('relates-to');
  });

  it('normalizes "contradicts" synonyms', () => {
    expect(normalizeEdgeType('contradicts').canonical).toBe('contradicts');
    expect(normalizeEdgeType('contradiction').canonical).toBe('contradicts');
  });

  it('normalizes "supersedes" synonyms', () => {
    expect(normalizeEdgeType('supersedes').canonical).toBe('supersedes');
    expect(normalizeEdgeType('supersede').canonical).toBe('supersedes');
  });

  it('normalizes "used-by" synonyms', () => {
    expect(normalizeEdgeType('used-by').canonical).toBe('used-by');
    expect(normalizeEdgeType('used').canonical).toBe('used-by');
  });

  it('normalizes "implements" synonyms', () => {
    expect(normalizeEdgeType('implements').canonical).toBe('implements');
    expect(normalizeEdgeType('implementation').canonical).toBe('implements');
  });

  it('normalizes "causes" synonyms', () => {
    expect(normalizeEdgeType('causes').canonical).toBe('causes');
    expect(normalizeEdgeType('causality').canonical).toBe('causes');
  });

  it('passes through unknown edge types (NM-2)', () => {
    const result = normalizeEdgeType('custom-type');
    expect(result.canonical).toBe('custom-type');
    expect(result.invert).toBe(false);
  });
});

describe('NM-1: inverse edge direction', () => {
  it('blocked-by inverts (swaps endpoints)', () => {
    const result = normalizeEdgeType('blocked-by');
    expect(result.canonical).toBe('blocks');
    expect(result.invert).toBe(true);
  });

  it('superseded-by inverts', () => {
    const result = normalizeEdgeType('superseded-by');
    expect(result.canonical).toBe('supersedes');
    expect(result.invert).toBe(true);
  });

  it('superseded inverts', () => {
    const result = normalizeEdgeType('superseded');
    expect(result.canonical).toBe('supersedes');
    expect(result.invert).toBe(true);
  });

  it('uses inverts (A uses B → B used-by A)', () => {
    const result = normalizeEdgeType('uses');
    expect(result.canonical).toBe('used-by');
    expect(result.invert).toBe(true);
  });

  it('caused-by inverts (A caused-by B → B causes A)', () => {
    const result = normalizeEdgeType('caused-by');
    expect(result.canonical).toBe('causes');
    expect(result.invert).toBe(true);
  });

  it('non-inverse types do not swap', () => {
    expect(normalizeEdgeType('blocks').invert).toBe(false);
    expect(normalizeEdgeType('depends-on').invert).toBe(false);
    expect(normalizeEdgeType('relates-to').invert).toBe(false);
    expect(normalizeEdgeType('contradicts').invert).toBe(false);
    expect(normalizeEdgeType('supersedes').invert).toBe(false);
    expect(normalizeEdgeType('used-by').invert).toBe(false);
    expect(normalizeEdgeType('implements').invert).toBe(false);
    expect(normalizeEdgeType('causes').invert).toBe(false);
  });
});

describe('isValidCategory', () => {
  it('accepts valid categories', () => {
    for (const cat of VALID_CATEGORIES) {
      expect(isValidCategory(cat)).toBe(true);
    }
  });

  it('rejects invalid categories', () => {
    expect(isValidCategory('invalid')).toBe(false);
    expect(isValidCategory('')).toBe(false);
  });
});

describe('isCanonicalEdgeType', () => {
  it('accepts canonical types', () => {
    for (const type of CANONICAL_EDGE_TYPES) {
      expect(isCanonicalEdgeType(type)).toBe(true);
    }
  });

  it('rejects non-canonical types', () => {
    expect(isCanonicalEdgeType('custom-type')).toBe(false);
  });
});
