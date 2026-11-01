// M0 S0.1 T1 — the manifests (contracts §6–§8; design 07 §7). modeler.toml as ttr-core reads it (the reader
// check-model and the editor use); connections.toml and toolchain.lock as their readers read them.
//
// Run: just verify-model   (or: TATRMAN_PACKAGES=<ttr-core>/packages node --test model/tests/manifest.test.mjs)

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { repoRoot, tatrmanPackages } from './project-harness.mjs';

const { parseManifest, resolveManifest } = await import(
  pathToFileURL(path.join(tatrmanPackages, 'semantics/dist/index.js')).href
);
const manifest = resolveManifest(
  parseManifest(await readFile(path.join(repoRoot, 'modeler.toml'), 'utf-8')),
  repoRoot + '/',
);

test('M1 — modeler.toml: project hartland, Czech preferred, no manifest diagnostics', () => {
  assert.equal(manifest.name, 'hartland');
  assert.equal(manifest.preferredLanguage, 'cs');
  assert.deepEqual(manifest.configDiagnostics, []);
});

test('M2 — the declared kinds are db, er, md, binding, lexicon (cn joins at M4)', () => {
  assert.deepEqual([...manifest.declaredSchemas].sort(), ['binding', 'db', 'er', 'lexicon', 'md']);
});

test('M3 — one schema handle `hartland`: DB hartland, Postgres schema hartland (contracts §8), the default', () => {
  assert.deepEqual(Object.keys(manifest.schemas), ['hartland']);
  const h = manifest.schemas.hartland;
  assert.equal(h.database, 'hartland');
  assert.equal(h.dbSchema, 'hartland');
  assert.equal(h.dialect, 'postgres');
  assert.equal(manifest.defaults.schema, 'hartland');
});

test('M4 — one flat package: layout off, every model file declares `package hartland`', async () => {
  assert.equal(manifest.packages.layout, 'off');
  const { modelFiles } = await import('./project-harness.mjs');
  for (const file of await modelFiles()) {
    const text = await readFile(file, 'utf-8');
    assert.match(text, /^package hartland$/m, `${path.relative(repoRoot, file)} declares package hartland`);
  }
});

test('M5 — connections.toml: the one descriptor pg-hartland (CZK, cs, the OpenMetadata locators)', async () => {
  const text = await readFile(path.join(repoRoot, 'model/connections.toml'), 'utf-8');
  assert.equal((text.match(/^\[\[connection\]\]$/gm) ?? []).length, 1);
  for (const line of [
    'id       = "pg-hartland"',
    'database = "hartland"',
    'dialect  = "postgres"',
    'currency = "CZK"',
    'locale   = "cs"',
    'openmetadata.service = "hartland-pg"',
    'openmetadata.schema = "hartland"',
  ]) {
    assert.ok(text.includes(line), `connections.toml has ${line}`);
  }
  assert.match(text, /^schema\s+= "hartland"/m);
});

test('M6 — toolchain.lock reads as four pins (scripts/toolchain-lock.py): ttr-core 0.20.x and the metadata loader', () => {
  const out = execFileSync('python3', [path.join(repoRoot, 'scripts/toolchain-lock.py')], { encoding: 'utf-8' });
  const pins = Object.fromEntries(out.trim().split('\n').map((l) => l.split('=')));
  assert.deepEqual(Object.keys(pins), ['TTR_CORE_REF', 'VELES_IMAGE', 'VELES_TTR_METADATA', 'STRICT_LOAD']);
  assert.match(pins.TTR_CORE_REF, /^grammar\/v0\.20\.\d+$/);
  assert.equal(pins.STRICT_LOAD, 'true');
});
