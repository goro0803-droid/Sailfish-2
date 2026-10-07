// 夜間の処理を止める（タスクスケジューラが毎朝7時に起動する）。runner と、その子プロセス（学習・エンジンなど）をまとめて終了する。
// 各処理は途中結果を保存しているので、次の晩に続きから再開する。
import { execFileSync } from 'node:child_process';
import { readFileSync, appendFileSync, existsSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';

const DATA = 'D:\\shogi_data\\night';
const PID = join(DATA, 'runner.pid');
const LOG = join(DATA, `${new Date().toLocaleDateString('sv-SE')}.log`);
const log = s => appendFileSync(LOG, `${new Date().toLocaleString('ja-JP')} ${s}\n`);
if (!existsSync(PID)) { log('停止: runner は動いていない'); process.exit(0); }
const pid = readFileSync(PID, 'utf8').trim();
// PID は再利用されるので、本当に night/runner.mjs を実行している node かを確かめてから止める（無関係なプロセスを止めない）
let cmdline = '';
try {
  cmdline = execFileSync('powershell.exe', ['-NoProfile', '-Command',
    `(Get-CimInstance Win32_Process -Filter "ProcessId=${Number(pid)}").CommandLine`], { encoding: 'utf8', windowsHide: true });
} catch { /* 取得できなければ止めない */ }
if (!/node/i.test(cmdline) || !/night[\\/]runner\.mjs/i.test(cmdline)) {
  log(`停止: pid ${pid} は runner ではない（既に終了している）ので何もしない`);
  unlinkSync(PID);
  process.exit(0);
}
try {
  execFileSync('taskkill.exe', ['/PID', pid, '/T', '/F'], { stdio: 'ignore', windowsHide: true });
  log(`停止: runner（pid ${pid}）と子プロセスを終了した`);
} catch { log(`停止: runner（pid ${pid}）は既に終了していた`); }
unlinkSync(PID);
