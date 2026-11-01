// M0 S0.1 T3 — the db layer (design 07 §1, contracts §1, ⚑NH-P-10). Every table of data/ddl/schema.sql — the
// inventory's rendering — declared once, in the file of its inventory group; the views in views.ttrm; every column
// typed; the foreign keys in fks.ttrm; the whole model resolving clean. The column-by-column comparison with
// schema.sql is `just ddl-drift` (scripts/ddl_drift.py); this suite checks the layer's shape.
//
// Run: just verify-model

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { repoRoot, modelFiles, parseString, loadProject, blocking } from '../../tests/project-harness.mjs';

const schemaSql = await readFile(path.join(repoRoot, 'data/ddl/schema.sql'), 'utf-8');

// schema.sql's own roster: each CREATE TABLE under the `-- ── <group> ──` header before it; the views.
const expectedTables = new Map(); // table -> group
const expectedViews = new Map(); // view -> 'table' (materialized) | 'view'
{
  let group = null;
  for (const line of schemaSql.split('\n')) {
    const header = line.match(/^-- ── (\w+) ─/);
    if (header) group = header[1];
    const table = line.match(/^CREATE TABLE hartland\.(\w+) \(/);
    if (table) expectedTables.set(table[1], group);
    const view = line.match(/^CREATE (MATERIALIZED )?VIEW hartland\.(\w+) AS/);
    if (view) expectedViews.set(view[2], view[1] ? 'table' : 'view');
  }
}

const parsed = new Map(); // relative path -> ast
for (const file of await modelFiles('db')) {
  const result = parseString(await readFile(file, 'utf-8'), `file://${file}`);
  assert.deepEqual(result.errors, [], `${path.basename(file)} parses: ${JSON.stringify(result.errors)}`);
  parsed.set(path.relative(path.join(repoRoot, 'model/db'), file), result.ast);
}
const defs = [...parsed].flatMap(([file, ast]) => ast.definitions.map((d) => ({ file, def: d })));
const ttrmText = new Map();
for (const file of parsed.keys()) ttrmText.set(file, await readFile(path.join(repoRoot, 'model/db', file), 'utf-8'));

test('D1 — schema.sql holds 98 tables and 5 views (the inventory, contracts §1)', () => {
  assert.equal(expectedTables.size, 98);
  assert.equal(expectedViews.size, 5);
});

test('D2 — every model/db file is `model db schema hartland` in `package hartland`', () => {
  for (const [file, ast] of parsed) {
    assert.equal(ast.modelDirective?.modelCode, 'db', file);
    assert.equal(ast.modelDirective?.schema, 'hartland', file);
    assert.equal(ast.packageDecl?.name, 'hartland', file);
  }
});

test('D3 — every table is declared once, in <its inventory group>.ttrm; nothing else is', () => {
  const tables = defs.filter((d) => d.def.kind === 'table');
  const seen = new Map();
  for (const { file, def } of tables) {
    assert.ok(!seen.has(def.name), `${def.name} declared twice (${seen.get(def.name)}, ${file})`);
    seen.set(def.name, file);
  }
  for (const [name, group] of expectedTables) {
    assert.equal(seen.get(name), `${group}.ttrm`, `${name} lives in ${group}.ttrm`);
  }
  const materialized = [...expectedViews].filter(([, k]) => k === 'table').map(([v]) => v);
  const extra = [...seen.keys()].filter((t) => !expectedTables.has(t) && !materialized.includes(t));
  assert.deepEqual(extra, [], 'no table the inventory does not have');
});

test('D4 — every table column is typed; every table has a primary key whose columns are keys (views: none)', () => {
  for (const { file, def } of defs.filter((d) => d.def.kind === 'table')) {
    assert.ok(def.columns.length > 0, `${def.name} has columns`);
    for (const c of def.columns) assert.ok(c.type, `${def.name}.${c.name} is typed`);
    if (file === 'views.ttrm') continue;
    assert.ok(def.primaryKey?.length > 0, `${def.name} has a primary key`);
    const keys = def.columns.filter((c) => c.isKey).map((c) => c.name);
    assert.deepEqual(keys, def.primaryKey, `${def.name}: isKey marks exactly the primary key`);
  }
});

test('D5 — views.ttrm: the four materialized views as `def table` (⚑NH-P-10), return_request as `def view` + SQL', () => {
  const inViews = defs.filter((d) => d.file === 'views.ttrm');
  const text = ttrmText.get('views.ttrm');
  for (const [name, kind] of expectedViews) {
    const d = inViews.find((x) => x.def.name === name);
    assert.ok(d, `${name} is in views.ttrm`);
    assert.equal(d.def.kind, kind, `${name} is a def ${kind}`);
    if (kind === 'table') {
      const at = text.indexOf(`def table ${name} {`);
      const before = text.slice(text.lastIndexOf('\n\n', at), at);
      assert.match(before, /\/\/ materialized view — refreshed by load-world and prolong/, `${name}'s leading comment`);
    } else {
      assert.match(d.def.definitionSql?.value ?? '', /FROM return_request_hdr h/);
    }
  }
  assert.equal(inViews.length, expectedViews.size);
});

test('D6 — fks.ttrm holds every foreign key of schema.sql (205), and no other file holds one', () => {
  const fks = defs.filter((d) => d.def.kind === 'fk');
  assert.ok(fks.every((d) => d.file === 'fks.ttrm'), 'foreign keys live in fks.ttrm');
  const sqlFks = [...schemaSql.matchAll(/CONSTRAINT (\w+) FOREIGN KEY/g)].map((m) => m[1]).sort();
  assert.equal(sqlFks.length, 205);
  assert.deepEqual(fks.map((d) => d.def.name).sort(), sqlFks);
});

test('D7 — the whole model resolves: no error or warning beyond the accepted residuals', async () => {
  const project = await loadProject();
  for (const [file, errors] of project.parseErrors) assert.deepEqual(errors, [], `${file} parses`);
  assert.deepEqual(blocking(project.diagnostics), []);
});
