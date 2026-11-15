#!/usr/bin/env node

// Check that task prompts describe only the dataset, never the seed paper.
//
// A prompt is the system's sole input. Anything pointing back at the seed -- its
// research question, its methods, its DOI, or the history of how the anchor was
// chosen -- leaks the answer and makes Alignment scoring meaningless.
//
// Usage:
//   node benchmark/scripts/audit_prompt_contract.mjs

import fs from "node:fs/promises";
import path from "node:path";

const BENCHMARK = path.resolve(import.meta.dirname, "..");
const INPUT_DIR = path.join(BENCHMARK, "dataset", "task_inputs");

// Fields that exist only to record audit provenance and must not ship in a prompt.
const BANNED_FIELDS = ["region_scope_provenance", "asset_provenance"];

// Wording that betrays knowledge of the source paper.
const BANNED_PATTERNS = [
  [/\bseed\b/i, "refers to the seed paper"],
  [/full[- ]text/i, "refers to the source full text"],
  [/\bpaper_ideas?\b/i, "cites the paper_ideas file"],
  [/\breference_sets?\b/i, "cites the reference_sets file"],
  [/10\.\d{4,9}\//, "contains a DOI"],
  [/research[- ]question/i, "cites the seed research question"],
  [/methods? section/i, "cites a methods section"],
  [/study area section/i, "cites a study-area section"],
  [/\brepointed\b/i, "records anchor change history"],
  [/\bsubstituted[_ ]for\b/i, "records a substitution relationship"],
  [/former anchor/i, "records anchor change history"],
  [/\bthe authors?\b/i, "refers to the paper's authors"],
];

const walk = (value, trail, out) => {
  if (typeof value === "string") {
    out.push([trail.join("."), value]);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((v, index) => walk(v, [...trail, String(index)], out));
    return;
  }
  if (value && typeof value === "object") {
    for (const [key, v] of Object.entries(value)) walk(v, [...trail, key], out);
  }
};

const main = async () => {
  const tasks = (await fs.readdir(INPUT_DIR, { withFileTypes: true }))
    .filter((e) => e.isDirectory() && /^task_\d{3}$/.test(e.name))
    .map((e) => e.name)
    .sort();

  const findings = [];
  for (const taskId of tasks) {
    const prompt = JSON.parse(await fs.readFile(path.join(INPUT_DIR, taskId, "prompt.json"), "utf8"));

    for (const field of BANNED_FIELDS) {
      if (field in (prompt.data_description ?? {})) {
        findings.push({
          task_id: taskId, code: "BANNED_FIELD", field,
          message: `data_description.${field} is audit provenance and belongs in reference_sets or paper_ideas`,
        });
      }
    }

    const strings = [];
    walk(prompt, [], strings);
    for (const [where, text] of strings) {
      for (const [pattern, why] of BANNED_PATTERNS) {
        const hit = pattern.exec(text);
        if (hit) {
          findings.push({
            task_id: taskId, code: "SEED_REFERENCE", field: where,
            matched: hit[0], message: why,
            excerpt: text.length > 120 ? `${text.slice(0, 120)}...` : text,
          });
          break;
        }
      }
    }
  }

  const byTask = new Set(findings.map((f) => f.task_id));
  process.stdout.write(`${JSON.stringify({
    status: findings.length ? "violations_found" : "clean",
    tasks_checked: tasks.length,
    tasks_with_violations: byTask.size,
    finding_count: findings.length,
    findings,
  }, null, 2)}\n`);
  process.exitCode = findings.length ? 1 : 0;
};

await main();
