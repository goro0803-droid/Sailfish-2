// 夜間の処理を順に実行する（タスクスケジューラが毎晩23時に起動する。7時に night/stop.mjs が止める）
// - night/jobs.json に並べた処理を上から順に実行し、終わったものは D:\shogi_data\night\state.json に記録して次の晩は飛ばす。
// - 各処理は「止まっても次に続きから再開できる」作りにしておくこと（7時に強制終了されるため）。
// - 処理に "exclusive": true を付けたときだけ、floodgate の Sailfish（将棋アリーナ配下の sailfish2.exe）が止まるまで待つ
//   （スレッド数の比較など、CPUを占有しないと正しく測れない処理のため）。ほかの処理は floodgate と並行して動かしてよい（ユーザー了承済み）。
// - 23時〜7時の外では新しい処理を始めない。処理が失敗したら、そこで止める（後の処理が前提にしている可能性があるため）。
import { spawn, execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, appendFileSync, mkdirSync, existsSync, unlinkSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');                 // Sailfish2 フォルダ
const DATA = 'D:\\shogi_data\\night';
mkdirSync(DATA, { recursive: true });
const STATE = join(DATA, 'state.json');
const day = new Date().toLocaleDateString('sv-SE');
const LOG = join(DATA, `${day}.log`);
const log = s => appendFileSync(LOG, `${new Date().toLocaleString('ja-JP')} ${s}\n`);
const sleep = ms => new Promise(r => setTimeout(r, ms));
// --now を付けると時間帯の制限を外す（日中に手で実行するとき。ユーザーは floodgate と並行して動かすことを了承済み）
const NOW = process.argv.includes('--now');
const inWindow = () => { const h = new Date().getHours(); return NOW || h >= 23 || h < 7; };

function floodgateRunning() {
  try {
    const out = execFileSync('powershell.exe', ['-NoProfile', '-Command',
      "Get-CimInstance Win32_Process -Filter \"Name='sailfish2.exe'\" | ForEach-Object { $_.ExecutablePath }"], { encoding: 'utf8', windowsHide: true });
    return out.split(/\r?\n/).some(p => /ShogiArena/i.test(p));
  } catch { return false; }
}

// 別の runner が動いていたら何もしない（日中に --now で始めた分が23時まで続いた場合など。二重実行を防ぐ）
const PIDFILE = join(DATA, 'runner.pid');
if (existsSync(PIDFILE)) {
  const other = readFileSync(PIDFILE, 'utf8').trim();
  let cmdline = '';
  try {
    cmdline = execFileSync('powershell.exe', ['-NoProfile', '-Command', `(Get-CimInstance Win32_Process -Filter "ProcessId=${Number(other)}").CommandLine`], { encoding: 'utf8', windowsHide: true });
  } catch { /* 取得できなければ、動いていないとみなす */ }
  if (/node/i.test(cmdline) && /night[\\/]runner\.mjs/i.test(cmdline)) { log(`別の runner（pid ${other}）が動いているので、何もしない`); process.exit(0); }
}
writeFileSync(PIDFILE, String(process.pid));
const state = existsSync(STATE) ? JSON.parse(readFileSync(STATE, 'utf8')) : { done: {} };
const jobs = JSON.parse(readFileSync(join(HERE, 'jobs.json'), 'utf8'));
log(`開始（pid ${process.pid}）。未完了の処理: ${jobs.filter(j => !state.done[j.id]).map(j => j.id).join(', ') || 'なし'}`);

for (const job of jobs) {
  if (state.done[job.id]) continue;
  if (!inWindow()) { log('7時を過ぎたので、新しい処理は始めない'); break; }
  if (job.exclusive) {
    let stop = false;
    while (floodgateRunning()) {
      if (!inWindow()) { stop = true; break; }
      log(`floodgate の Sailfish が動いているので、${job.id} の開始を5分待つ`);
      await sleep(5 * 60 * 1000);
    }
    if (stop) { log('floodgate が動いたまま 7時を過ぎたので、ここで止める'); break; }
  }
  log(`処理開始: ${job.id}（${job.name}）`);
  const code = await new Promise(res => {
    const p = spawn(job.cmd[0], job.cmd.slice(1), { cwd: resolve(ROOT, job.cwd || '.'), windowsHide: true });
    p.stdout.on('data', d => appendFileSync(LOG, d));
    p.stderr.on('data', d => appendFileSync(LOG, d));
    p.on('close', res);
    p.on('error', e => { log(`起動できません: ${e.message}`); res(-1); });
  });
  if (code !== 0) { log(`処理が失敗（終了コード ${code}）: ${job.id}。ここで止める`); break; }
  state.done[job.id] = new Date().toISOString();
  writeFileSync(STATE, JSON.stringify(state, null, 1));
  log(`処理完了: ${job.id}`);
}
log('終了');
try { unlinkSync(join(DATA, 'runner.pid')); } catch { /* 既にない */ }
