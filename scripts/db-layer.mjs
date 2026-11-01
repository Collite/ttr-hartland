// The db layer as JSON — model/db/*.ttrm read by ttr-core's own parser (the toolchain the model-gate builds), for
// scripts/ddl_drift.py to compare with data/ddl/schema.sql (M0 S0.1 T4, contracts §1). generated/resolved-packages.json
// lists packages and entities, not tables, so the drift check reads the files.
//
//   TATRMAN_PACKAGES=<ttr-core>/packages node scripts/db-layer.mjs [<db-dir>]   # default: ../tatrman/packages, model/db
//
// Prints { tables: {name: {file, kind, primaryKey, columns: [{name, type, length, precision, optional}]}},
//          fks: {name: {file, from: ["table.column"], to: ["table.column"]}}, parseErrors: ["file: message"] }.
// Exit 0 even with parse errors: they are part of the report, and the caller fails on them.

import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packages = process.env.TATRMAN_PACKAGES
  ? path.resolve(process.env.TATRMAN_PACKAGES)
  : path.resolve(repoRoot, '../tatrman/packages');
const { parseString } = await import(pathToFileURL(path.join(packages, 'parser/dist/index.js')).href);

const dbDir = process.argv[2] ? path.resolve(process.argv[2]) : path.join(repoRoot, 'model/db');
const files = (await readdir(dbDir)).filter((f) => f.endsWith('.ttrm')).sort();

const type = (t) =>
  t?.kind === 'structured'
    ? { type: t.typeName, length: t.length ?? null, precision: t.precision ?? null }
    : { type: t?.name ?? null, length: null, precision: null };

// `db.hartland.<table>.<column>` (or a relative `<table>.<column>`) → `<table>.<column>`
const columnRef = (item) => (item.parts ?? item.path.split('.')).slice(-2).join('.');

const out = { tables: {}, fks: {}, parseErrors: [] };
for (const file of files) {
  const result = parseString(await readFile(path.join(dbDir, file), 'utf-8'), `file://${path.join(dbDir, file)}`);
  for (const e of result.errors.filter((e) => e.severity === 'error')) {
    out.parseErrors.push(`${file}:${e.source?.line ?? '?'}: ${e.message}`);
  }
  for (const def of result.ast?.definitions ?? []) {
    if (def.kind === 'table' || def.kind === 'view') {
      if (out.tables[def.name]) out.parseErrors.push(`${file}: ${def.name} is declared twice`);
      out.tables[def.name] = {
        file,
        kind: def.kind,
        primaryKey: def.primaryKey ?? [],
        columns: (def.columns ?? []).map((c) => ({ name: c.name, ...type(c.type), optional: c.optional === true })),
      };
    } else if (def.kind === 'fk') {
      out.fks[def.name] = { file, from: (def.from?.items ?? []).map(columnRef), to: (def.to?.items ?? []).map(columnRef) };
    }
  }
}
console.log(JSON.stringify(out, null, 1));
