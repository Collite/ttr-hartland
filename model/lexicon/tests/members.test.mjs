// M0 S0.1 T5 — the member word lists (design 07 §5; scripts/seed_members.py) and the member vocabularies' split on
// the db columns (MV: names partial — TYPOS/TOKENS — codes EXACT). The seed's own drift, size and one-word-one-member
// checks are pytest's (data/sim/tests/test_members.py); this suite reads the files as the toolchain does.
//
// Run: just verify-model

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { repoRoot, modelFiles, parseString } from '../../tests/project-harness.mjs';

const lists = new Map(); // stem -> ast
for (const file of await modelFiles('lexicon/cs/members')) {
  const result = parseString(await readFile(file, 'utf-8'), `file://${file}`);
  assert.deepEqual(result.errors, [], `${path.basename(file)} parses`);
  lists.set(path.basename(file, '.ttrm'), result.ast);
}

test('L1 — the nine lists, each `model lexicon locale cs`, every entry a term with a `for:` and its forms', () => {
  assert.deepEqual(
    [...lists.keys()].sort(),
    ['brand', 'chain', 'kraj', 'okres', 'pack', 'product', 'product_line', 'site', 'sku'],
  );
  for (const [stem, ast] of lists) {
    assert.equal(ast.modelDirective?.modelCode, 'lexicon', stem);
    assert.equal(ast.modelDirective?.locale, 'cs', stem);
    for (const def of ast.definitions) {
      assert.equal(def.kind, 'term', `${stem}: ${def.name}`);
      assert.match(def.name, new RegExp(`^${stem}_[a-z0-9_]+_cs$`), `${stem}: ${def.name}`);
      assert.ok(def.forms?.length > 0, `${stem}: ${def.name} has forms`);
    }
  }
});

test('L2 — every term targets its vocabulary column (the placeholder until M3 S3.1 T5)', () => {
  for (const [stem, ast] of lists) {
    const want = stem === 'pack' ? 'db.hartland.sku.pack_format' : `db.hartland.${stem}.name`;
    for (const def of ast.definitions) assert.equal(def.target?.path, want, `${stem}: ${def.name}`);
  }
});

test('L3 — the member vocabularies on the db columns: names partial (TYPOS/TOKENS), codes EXACT (MV)', async () => {
  const split = {
    sku: ['sku_code', 'name'], product: ['code', 'name'], product_line: ['code', 'name'], brand: ['code', 'name'],
    chain: ['code', 'name'], kraj: ['code', 'name'], okres: ['code', 'name'], site: ['code', 'name'],
    carrier: ['code', 'name'],
  };
  const tables = new Map();
  for (const file of await modelFiles('db')) {
    const { ast } = parseString(await readFile(file, 'utf-8'), `file://${file}`);
    for (const def of ast.definitions) if (def.kind === 'table') tables.set(def.name, def);
  }
  for (const [table, [code, name]] of Object.entries(split)) {
    const cols = Object.fromEntries(tables.get(table).columns.map((c) => [c.name, c]));
    assert.equal(cols[code].search?.method?.name, 'EXACT', `${table}.${code} is indexed EXACT`);
    assert.match(cols[name].search?.method?.name ?? '', /^(TYPOS|TOKENS)$/, `${table}.${name} is partial`);
  }
});

test('L4 — Czech only: no en/ locale (NH-D8), and the cs README says so', async () => {
  assert.ok(!existsSync(path.join(repoRoot, 'model/lexicon/en')), 'model/lexicon/en does not exist');
  const entries = await readdir(path.join(repoRoot, 'model/lexicon'));
  assert.deepEqual(entries.filter((e) => !['cs', 'tests'].includes(e)), []);
  const readme = await readFile(path.join(repoRoot, 'model/lexicon/cs/README.md'), 'utf-8');
  assert.match(readme, /the `en` locale is not authored \(NH-D8\)/);
});
