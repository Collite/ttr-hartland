// The project-load harness of the model suites (`just verify-model`; ttr-demo's model/tests/project-harness.mjs,
// trimmed). Parses every model/**/*.ttrm as ONE project and runs ttr-core's semantics + lint pipeline over it — the
// machinery the editor runs — borrowing the built dist of a ttr-core checkout at toolchain.lock's [ttr-core].ref
// (this repo vendors no toolchain).
//
// Which checkout: TATRMAN_PACKAGES (its packages/ folder), else the sibling ../tatrman/packages — the layout CI builds
// (.github/actions/toolchain) and ttr-demo's `cli=` defaults assume.

import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
export const repoRoot = path.resolve(here, '../..');
export const tatrmanPackages = process.env.TATRMAN_PACKAGES
  ? path.resolve(process.env.TATRMAN_PACKAGES) // relative to the working directory, as a shell user means it
  : path.resolve(repoRoot, '../tatrman/packages');
const dist = (pkg, file) => import(pathToFileURL(path.join(tatrmanPackages, pkg, 'dist', file)).href);

const { parseString } = await dist('parser', 'index.js');
const semantics = await dist('semantics', 'index.js');
const semanticsNodeOnly = await dist('semantics', 'node-only.js');
const lintMod = await dist('lint', 'index.js');

const { resolveManifest, parseManifest, ProjectSymbolTable, Resolver, PackageGraphBuilder, synthesizeMappings, effectivePackage } =
  semantics;
const { loadStockVocabularies } = semanticsNodeOnly;
const { lintDocument, lintProject, recommendedConfig } = lintMod;

// Diagnostics the project sweep accepts — each with the reason. A new entry needs its stage and why.
export const ACCEPTED_RESIDUAL_CODES = new Set([
  // M0 T1: one flat `package hartland` in a kind-directory tree (modeler.toml [packages]). Info-level and
  // non-suppressible by design; resolution is unaffected.
  'ttr/package-prefix-divergence',
]);

export { parseString };

async function walkTtrm(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (['node_modules', '.git'].includes(entry.name)) continue;
      out.push(...(await walkTtrm(path.join(dir, entry.name))));
    } else if (entry.isFile() && entry.name.endsWith('.ttrm')) {
      out.push(path.join(dir, entry.name));
    }
  }
  return out.sort();
}

/** Every model/**\/*.ttrm path, absolute, sorted. */
export async function modelFiles(sub = '') {
  return walkTtrm(path.join(repoRoot, 'model', sub));
}

/**
 * Parse + resolve + lint the whole model/ tree as one project. Returns per-file ASTs, parse errors and lint
 * diagnostics ({ code, severity, message, line }), the symbol table and the manifest.
 */
export async function loadProject() {
  const root = repoRoot.endsWith('/') ? repoRoot : repoRoot + '/';
  const files = await modelFiles();
  const manifest = resolveManifest(parseManifest(await readFile(path.join(repoRoot, 'modeler.toml'), 'utf-8')), root);

  const symbols = new ProjectSymbolTable();
  const asts = new Map();
  for (const [name, ast] of await loadStockVocabularies(manifest.stockVocabularies)) {
    const uri = `stock://${name}.ttrm`;
    asts.set(uri, ast);
    symbols.upsertDocument(uri, ast, 'cnc', 'role', '');
  }

  const parseErrors = new Map();
  for (const file of files) {
    const uri = `file://${file}`;
    const result = parseString(await readFile(file, 'utf-8'), uri);
    parseErrors.set(path.relative(repoRoot, file), result.errors);
    if (!result.ast) continue;
    asts.set(uri, result.ast);
    symbols.upsertDocument(
      uri,
      result.ast,
      result.ast.modelDirective?.modelCode ?? 'db',
      result.ast.modelDirective?.schema ?? '',
      effectivePackage(result.ast, file, root, manifest.packages),
    );
    synthesizeMappings(symbols, uri, result.ast);
  }

  const resolver = new Resolver(symbols);
  const deps = { manifest, symbols, resolver };
  const packageGraph = new PackageGraphBuilder(symbols, asts).build();
  const config = recommendedConfig();
  const projectByUri = lintProject(asts, packageGraph, deps, config);

  const diagnostics = new Map();
  for (const file of files) {
    const uri = `file://${file}`;
    const ast = asts.get(uri);
    if (!ast) continue;
    const all = [...lintDocument(uri, ast, deps, config), ...(projectByUri.get(uri) ?? [])];
    diagnostics.set(
      path.relative(repoRoot, file),
      all.map((d) => ({ code: d.code, severity: d.severity, message: d.message, line: d.source?.line })),
    );
  }
  return { files, asts, symbols, resolver, manifest, parseErrors, diagnostics };
}

/** The diagnostics a gate fails on: every error or warning whose code is not an accepted residual, per file. */
export function blocking(diagnostics) {
  const out = [];
  for (const [file, list] of diagnostics) {
    for (const d of list) {
      if ((d.severity === 'error' || d.severity === 'warning') && !ACCEPTED_RESIDUAL_CODES.has(d.code)) {
        out.push(`${file}:${d.line ?? '?'} ${d.severity} ${d.code} — ${d.message}`);
      }
    }
  }
  return out;
}
