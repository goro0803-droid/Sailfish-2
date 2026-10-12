// 複数の対局（tools/run-match-chunks.mjs）を同時に実行する。CPU の取り合いの影響を、比べる版どうしでそろえるため。
// 使い方: node tools/run-parallel-matches.mjs <設定.json>
// 設定.json: { "log": "ログのパス", "jobs": [ { "name": "名前", "args": [run-match-chunks.mjs の引数...] }, ... ] }
// 各対局は保存済みの区切りを飛ばすので、止まっても同じコマンドで続きから再開できる。
import { spawn } from 'node:child_process';
import { readFileSync, appendFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const cfg = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const log = s => appendFileSync(cfg.log, `${new Date().toLocaleString('ja-JP')} ${s}\n`);
log(`開始（pid ${process.pid}）: ${cfg.jobs.map(j => j.name).join(', ')}`);
const codes = await Promise.all(cfg.jobs.map(job => new Promise(res => {
  const p = spawn(process.execPath, ['--no-warnings', 'tools/run-match-chunks.mjs', ...job.args], { cwd: ROOT, windowsHide: true });
  p.stdout.on('data', d => appendFileSync(cfg.log, `[${job.name}] ${d}`));
  p.stderr.on('data', d => appendFileSync(cfg.log, `[${job.name}] ${d}`));
  p.on('close', code => { log(`${job.name} 終了（終了コード ${code}）`); res(code); });
  p.on('error', e => { log(`${job.name} 起動できません: ${e.message}`); res(-1); });
})));
log(`すべて終了: ${codes.join(', ')}`);
