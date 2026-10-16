// floodgate の日ごとの棋譜一覧のページから、指定の名前を含む .csa の URL を集める
// 使い方: node tools/list-floodgate-urls.mjs <名前> <出力 txt> <YYYY/MM/DD> [<YYYY/MM/DD> ...]
import { writeFileSync, existsSync, readFileSync } from 'node:fs';

const [name, out, ...days] = process.argv.slice(2);
const urls = new Set(existsSync(out) ? readFileSync(out, 'utf8').split(/\r?\n/).filter(Boolean) : []);
for (const day of days) {
  const base = `https://wdoor.c.u-tokyo.ac.jp/shogi/x/${day}/`;
  const res = await fetch(base);
  if (!res.ok) { console.log(`取得失敗 ${res.status} ${base}`); continue; }
  const html = await res.text();
  let n = 0;
  for (const m of html.matchAll(/href="([^"]+\.csa)"/g)) {
    if (m[1].includes(`+${name}+`)) { urls.add(new URL(m[1], base).href); n++; }
  }
  console.log(`${day}: ${n} 局`);
}
writeFileSync(out, [...urls].join('\n') + '\n');
