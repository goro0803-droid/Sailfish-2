from pathlib import Path
import subprocess,json,datetime,statistics,traceback,os
ROOT=Path(__file__).resolve().parent.parent
statepath=ROOT/'baseline/time_clock_state.json'
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def main():
    if statepath.exists():raise RuntimeError('Clock test already exists')
    state=dict(status='running',pid=os.getpid(),started=now(),games=20,initialMs=300000,incrementMs=10000,parallel=2,threads=1,seedOffset=40000)
    statepath.write_text(json.dumps(state,indent=2),encoding='utf-8')
    try:
        a='baseline/sailfish2_time_candidate.exe|Threads=1,NetworkDelay=120'
        b='baseline/sailfish2_before_time.exe|Threads=1,NetworkDelay=120'
        with (ROOT/'baseline/time_clock20.log').open('w',encoding='utf-8') as out:
            subprocess.run(['node','--no-warnings','tools/match-clock.mjs',a,b,'20','300000','10000','2','--seed-offset','40000','--json','baseline/time_clock20.json'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,check=True,timeout=86400)
        result=json.loads((ROOT/'baseline/time_clock20.json').read_text(encoding='utf-8'))
        banks=[[],[]];spent=[[],[]]
        for game in result['samples']:
            for engine in [0,1]:
                side=(0 if game['aBlack'] else 1) if engine==0 else (1 if game['aBlack'] else 0)
                moves=[m for m in game['trace'] if m['side']==side][:10]
                if len(moves)==10:
                    banks[engine].append(moves[-1]['afterMs']/1000)
                    spent[engine].append(sum(m['elapsedMs'] for m in moves)/1000)
        stats=dict(first10SpentMedian=[statistics.median(v) if v else None for v in spent],remainingAfter10Median=[statistics.median(v) if v else None for v in banks])
        state.update(status='complete',finished=now(),results=result['results'],score=result['score'],paired95=result['paired95'],timeStats=stats,reasons=result['reasons'])
        report=ROOT/'REPORT_時間配分改善_20261003.md'
        s=report.read_text(encoding='utf-8')+'\n## 時計対局の完了結果\n'+json.dumps({k:state[k] for k in ['results','score','paired95','timeStats','reasons']},ensure_ascii=False,indent=2)+'\n20局は初期比較で、勝率向上を確定する規模ではない。\n'
        report.write_text(s,encoding='utf-8')
    except Exception:
        state.update(status='failed',finished=now(),error=traceback.format_exc())
        raise
    finally:statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
