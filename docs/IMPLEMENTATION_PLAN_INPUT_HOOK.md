# Implementation Plan: KG Input Hook (Auto-Surface)

## Overview

Add a built-in `input` hook to the KG extension that automatically searches the knowledge graph when the user submits input, and surfaces high-scoring results into the system prompt. This is transparent to the agent — no agent configuration, no agent control.

**Goal:** Reduce the agent's work of searching the graph for relevant knowledge by pre-fetching it before each turn.

---

## Architecture

### Where it lives

- **Hook logic:** `hooks.ts` — new function `onInput()`
- **Subscription:** `index.ts` — new `pi.on('input', ...)` in `registerHooks()`
- **Config:** `search.ts` — new `inputSearchThreshold`, `inputSearchMaxResults`, `inputSearchTimeout` in `SearchConfig`

### How it works (data flow)

```
User submits input text
  → pi emits 'input' event (before skill/template expansion)
  → KG's built-in hook receives the event
  → Hook runs searchHybrid(inputText, maxResults, config)
  → If results have compositeScore >= inputSearchThreshold:
      → Format results as injection block
      → Inject into system prompt via pi.systemPrompt?.inject?.()
  → Agent turn continues with enriched context
```

### Injection format

When results are surfaced, they are injected as a system prompt block:

```
[KG Auto-Surface]
  - [category/subcategory] Content text (score: 0.72)
  - [category] Another content text (score: 0.68)
[End KG Auto-Surface]
```

If no results meet the threshold, the hook returns `null` and does nothing (no injection, no noise).

---

## File Changes

### 1. `search.ts` — Add config fields

Add three new fields to `SearchConfig`:

```typescript
export interface SearchConfig {
  // Existing fields (unchanged)
  embeddingEndpoint: string;
  embeddingModel: string;
  maxResults: number;
  ftxF5Weight: number;
  vectorWeight: number;
  frequencyWeight: number;

  // NEW: Input hook config
  inputSearchThreshold: number;    // Minimum compositeScore to surface (default: 0.65)
  inputSearchMaxResults: number;   // Cap on results for input hook (default: 3)
  inputSearchTimeout: number;      // ms before aborting (default: 2000)
}
```

Update `DEFAULT_CONFIG`:

```typescript
export const DEFAULT_CONFIG: SearchConfig = {
  // Existing defaults (unchanged)
  embeddingEndpoint: 'http://10.1.1.145:1234/v1/embeddings',
  embeddingModel: 'nomic-embed-text-v1.5',
  maxResults: 10,
  ftxF5Weight: 0.4,
  vectorWeight: 0.3,
  frequencyWeight: 0.3,

  // NEW defaults
  inputSearchThreshold: 0.65,
  inputSearchMaxResults: 3,
  inputSearchTimeout: 2000,
};
```

Update `validateConfig` to handle the new fields:

- Validate `inputSearchThreshold` is a number between 0 and 1 (clamp to range)
- Validate `inputSearchMaxResults` is a positive integer (default 3 if invalid)
- Validate `inputSearchTimeout` is a positive number (default 2000 if invalid)

### 2. `hooks.ts` — Add `onInput` function

Add a new exported function:

```typescript
export interface InputContext {
  text: string;
  images?: Array<{ type: string; source: any }>;
  source?: 'interactive' | 'rpc' | 'extension';
  streamingBehavior?: 'steer' | 'followUp' | undefined;
}
```

Implementation:

```typescript
import { searchHybrid } from './search.ts';
import { DEFAULT_CONFIG } from './search.ts';

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
```

Add a helper function:

```typescript
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
```

### 3. `index.ts` — Subscribe to `input` event

In `registerHooks()`, add a new subscription after the existing hooks:

```typescript
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
```

**Key points:**
- The subscription is wrapped in `try/catch` so hook failures don't break the agent turn.
- The injection is only performed if `onInput` returns a non-null string.
- The hook runs **before** the agent turn continues (same as all existing hooks).
- The agent has **no control** over this hook — no registration, no configuration, no visibility.

### 4. Export the new function from `index.ts`

Add `onInput` to the exports at the bottom of `index.ts`:

```typescript
export { onSessionStart, onBeforeAgentStart, onContext, onSessionBeforeCompact, onSessionShutdown, onInput };
```

---

## Unit Tests

### Test file: `test/input-hook.test.ts`

Create a new test file. The test framework is already set up (see existing `test/tools.test.ts` for patterns).

#### Test 1: Empty input returns null

```typescript
import { onInput } from '../hooks.js';
import { openKnowledgeGraph } from '../db.js';
import { DEFAULT_CONFIG } from '../search.js';
import path from 'path';

test('onInput returns null for empty text', () => {
  const db = openKnowledgeGraph(path.join(__dirname, 'test-kg-input.db'));
  
  // Test with empty string
  const result = onInput(db, { text: '' }, DEFAULT_CONFIG);
  expect(result).resolves.toBeNull();
  
  // Test with whitespace only
  const result2 = onInput(db, { text: '   ' }, DEFAULT_CONFIG);
  expect(result2).resolves.toBeNull();
  
  // Test with undefined text
  const result3 = onInput(db, { text: undefined }, DEFAULT_CONFIG);
  expect(result3).resolves.toBeNull();
  
  db.close();
});
```

#### Test 2: No matching nodes returns null

```typescript
test('onInput returns null when no nodes match', () => {
  const db = openKnowledgeGraph(path.join(__dirname, 'test-kg-input.db'));
  db.saveNode({
    category: 'knowledge',
    subcategory: 'fact',
    content: 'This is a test node about completely unrelated topics',
  });

  const result = onInput(db, { text: 'quantum computing and machine learning' }, DEFAULT_CONFIG);
  expect(result).resolves.toBeNull();
  db.close();
});
```

#### Test 3: High-scoring match returns injection block

```typescript
test('onInput returns injection block when nodes meet threshold', async () => {
  const db = openKnowledgeGraph(path.join(__dirname, 'test-kg-input.db'));
  
  db.saveNode({
    category: 'knowledge',
    subcategory: 'fact',
    content: 'Pi is a minimal terminal coding harness that runs in 4 modes.',
  });
  db.saveNode({
    category: 'knowledge',
    subcategory: 'fact',
    content: 'Built-in tools include read, write, edit, and bash.',
  });

  const result = await onInput(db, { text: 'pi terminal coding harness tools' }, DEFAULT_CONFIG);

  expect(result).not.toBeNull();
  expect(result).toContain('[KG Auto-Surface]');
  expect(result).toContain('[End KG Auto-Surface]');
  expect(result).toContain('[knowledge/fact]');
  expect(result).toContain('(score:');
  db.close();
});
```

#### Test 4: Results below threshold are filtered out

```typescript
test('onInput filters results below threshold', async () => {
  const db = openKnowledgeGraph(path.join(__dirname, 'test-kg-input.db'));
  
  db.saveNode({
    category: 'knowledge',
    subcategory: 'fact',
    content: 'This node has very low relevance to any reasonable query',
  });

  // Use a high threshold so the node won't qualify
  const config = { ...DEFAULT_CONFIG, inputSearchThreshold: 0.95 };
  const result = await onInput(db, { text: 'pi terminal coding harness tools' }, config);

  expect(result).toBeNull();
  db.close();
});
```

#### Test 5: Max results cap is respected

```typescript
test('onInput respects inputSearchMaxResults', async () => {
  const db = openKnowledgeGraph(path.join(__dirname, 'test-kg-input.db'));
  
  for (let i = 0; i < 10; i++) {
    db.saveNode({
      category: 'knowledge',
      subcategory: 'fact',
      content: `This is test node number ${i} about pi and coding`,
    });
  }

  const config = { ...DEFAULT_CONFIG, inputSearchMaxResults: 2 };
  const result = await onInput(db, { text: 'pi coding test' }, config);

  expect(result).not.toBeNull();
  // Count how many node entries are in the result
  const nodeMatches = result!.match(/\[knowledge\/fact\]/g);
  expect(nodeMatches).not.toBeNull();
  expect(nodeMatches!.length).toBeLessThanOrEqual(2);
  db.close();
});
```

#### Test 6: Injection format is correct

```typescript
test('onInput injection format matches spec', async () => {
  const db = openKnowledgeGraph(path.join(__dirname, 'test-kg-input.db'));
  
  db.saveNode({
    category: 'knowledge',
    subcategory: 'fact',
    content: 'Pi supports 30+ providers including Claude, OpenAI, and Google Gemini.',
  });

  const result = await onInput(db, { text: 'providers Claude OpenAI Gemini' }, DEFAULT_CONFIG);

  expect(result).not.toBeNull();
  expect(result).toMatch(/^\[KG Auto-Surface\]\n  - \[knowledge\/fact\] .+ \(score: \d+\.\d{3}\)\n\[End KG Auto-Surface\]$/m);
  db.close();
});
```

#### Test 7: Content is truncated to 200 chars

```typescript
test('onInput truncates content to 200 characters', async () => {
  const db = openKnowledgeGraph(path.join(__dirname, 'test-kg-input.db'));
  
  const longContent = 'x'.repeat(500);
  db.saveNode({
    category: 'knowledge',
    subcategory: 'fact',
    content: longContent,
  });

  const result = await onInput(db, { text: longContent.slice(0, 50) }, DEFAULT_CONFIG);

  expect(result).not.toBeNull();
  // The injected content should be at most 200 chars (excluding label and score)
  const contentMatch = result!.match(/\[knowledge\/fact\] (.+?) \(score:/);
  expect(contentMatch).not.toBeNull();
  expect(contentMatch![1].length).toBeLessThanOrEqual(200);
  db.close();
});
```

#### Test 8: Config validation handles invalid values

```typescript
import { validateConfig } from '../search.js';

test('validateConfig handles invalid inputSearchThreshold', () => {
  const result = validateConfig({ inputSearchThreshold: -0.5 });
  expect(result.inputSearchThreshold).toBeGreaterThanOrEqual(0);
  
  const result2 = validateConfig({ inputSearchThreshold: 1.5 });
  expect(result2.inputSearchThreshold).toBeLessThanOrEqual(1);
});

test('validateConfig handles invalid inputSearchMaxResults', () => {
  const result = validateConfig({ inputSearchMaxResults: -1 });
  expect(result.inputSearchMaxResults).toBe(3); // defaults back
});

test('validateConfig handles invalid inputSearchTimeout', () => {
  const result = validateConfig({ inputSearchTimeout: -100 });
  expect(result.inputSearchTimeout).toBe(2000); // defaults back
});
```

---

## Milestones

### Milestone 1: Config Changes (search.ts)

**Goal:** Add the three new config fields to `SearchConfig`, `DEFAULT_CONFIG`, and `validateConfig`.

**Checklist:**
- [ ] Add `inputSearchThreshold: number` to `SearchConfig` interface (default 0.65)
- [ ] Add `inputSearchMaxResults: number` to `SearchConfig` interface (default 3)
- [ ] Add `inputSearchTimeout: number` to `SearchConfig` interface (default 2000)
- [ ] Add all three to `DEFAULT_CONFIG`
- [ ] Update `validateConfig` to validate and clamp each field
- [ ] Run existing unit tests — no regressions

**Acceptance criteria:** All existing tests pass. `validateConfig` correctly handles invalid values for the new fields.

---

### Milestone 2: Hook Logic (hooks.ts)

**Goal:** Implement `onInput` and `formatInputSurface`.

**Checklist:**
- [ ] Add `InputContext` interface
- [ ] Implement `onInput(db, event, config)` function
- [ ] Implement `formatInputSurface(results)` helper
- [ ] Export `onInput` from module
- [ ] Unit tests for: empty input, no match, threshold filtering, max results cap, format correctness, content truncation, config validation

**Acceptance criteria:** All unit tests in `test/input-hook.test.ts` pass.

---

### Milestone 3: Integration (index.ts)

**Goal:** Subscribe the hook to the `input` event.

**Checklist:**
- [ ] Add `pi.on('input', ...)` subscription in `registerHooks()`
- [ ] Wrap in `try/catch` for fail-safety
- [ ] Call `onInput` with event fields: `text`, `images`, `source`, `streamingBehavior`
- [ ] Inject result via `pi.systemPrompt?.inject?.()`
- [ ] Export `onInput` from module
- [ ] No changes to tool registration or commands
- [ ] Verify existing hooks still work

**Acceptance criteria:** The extension loads without errors. The hook fires on user input and injects into the system prompt when results qualify.

---

### Milestone 4: End-to-End Testing

**Goal:** Verify the hook works in a real session.

**Checklist:**
- [ ] Start a session with the KG populated (≥ 5 nodes)
- [ ] Submit an input that should match existing nodes
- [ ] Verify the agent sees the auto-surfaced content in its context
- [ ] Submit an input that should NOT match (no injection)
- [ ] Verify no injection appears
- [ ] Verify the agent has no way to configure, disable, or inspect the hook

**Acceptance criteria:** The hook surfaces relevant knowledge automatically when the agent would otherwise need to search the graph manually.

---

## Configuration Reference

Users can configure the hook via `~/.pi/agent/settings.json`:

```json
{
  "kgMemory": {
    "inputSearchThreshold": 0.65,
    "inputSearchMaxResults": 3,
    "inputSearchTimeout": 2000
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `inputSearchThreshold` | number (0–1) | `0.65` | Minimum compositeScore to surface. Lower = more results surfaced. |
| `inputSearchMaxResults` | positive integer | `3` | Maximum number of results to search and surface. |
| `inputSearchTimeout` | positive number (ms) | `2000` | Timeout before aborting the search. |

---

## Edge Cases

1. **Empty input:** Returns `null`, no injection. (Handled by early return.)
2. **No matching nodes:** Returns `null`, no injection. (Handled by filter.)
3. **All nodes below threshold:** Returns `null`, no injection. (Handled by filter.)
4. **Embedding API unavailable:** Falls back to pure FTS5 (existing behavior).
5. **Hook throws an error:** Caught by `try/catch`, logged to console, no injection, agent turn continues.
6. **Very large graph:** `inputSearchMaxResults` caps the search. Timeout prevents stalling.
7. **Streaming input:** Hook receives `streamingBehavior` but processes the text the same way (no special streaming handling needed — the hook runs once on the input event).

---

## What This Does NOT Do

- The agent **cannot** register, configure, disable, or inspect this hook.
- The hook is **not** listed among the agent-visible tools.
- The hook does **not** add nodes to the graph (read-only).
- The hook does **not** modify the user's input text.
- The hook does **not** fire on agent-injected messages (only on raw user input).
