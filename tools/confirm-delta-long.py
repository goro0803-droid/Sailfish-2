from pathlib import Path
import subprocess, json, hashlib, math, statistics, datetime, traceback, os
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'baseline/delta_long_500ms_20261003'
A='target/release/sailfish2.exe|NetworkDelay=0,Threads=1'
B='baseline/sailfish2_before_delta.exe|NetworkDelay=0,Threads=1'
EXPECTED=['9ea053461cffbeac66475b76524588b3c424c12ee708c6c193f41cf3ed7a32ae','367cc5ede98c490301670b129eaa2c2ce1f1ec28b6e1ec6cac2a3d17277c2f13']
def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def save(data):
    tmp=OUT/'progress.tmp'
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(OUT/'progress.json')
def summarize(samples):
    vals=[x['score'] for x in samples]
    pairs=[sum(vals[i:i+2])/2 for i in range(0,len(vals)-1,2)]
    p=statistics.mean(vals) if vals else None
    interval=None
    if len(pairs)>1:
        err=1.96*statistics.stdev(pairs)/math.sqrt(len(pairs))
        interval=[max(0,p-err),min(1,p+err)]
    return dict(completed=len(vals),wins=vals.count(1),losses=vals.count(0),draws=vals.count(.5),score=p,paired95=interval)
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'progress.json').exists():
        raise RuntimeError('Existing run: refusing to overwrite')
    state=dict(status='running',started=now(),pid=os.getpid(),engineA=A,engineB=B,sha256=EXPECTED,games=600,moveMs=500,parallel=1,hashMB=64,threads=1,seedOffset=10000,priority='BELOW_NORMAL',samples=[])
    samples=[]
    save({**state,**summarize(samples)})
    try:
        for batch in range(30):
            if (OUT/'STOP_AFTER_BATCH').exists():
                state['status']='stopped_after_batch'
                break
            for spec,expected in zip([A,B],EXPECTED):
                actual=hashlib.sha256((ROOT/spec.split('|')[0]).read_bytes()).hexdigest()
                if actual!=expected:
                    raise RuntimeError('Engine changed during measurement: '+spec)
            dest=OUT/('batch_%02d.json'%batch)
            args=['node','tools/match.mjs',A,B,'20','500','1','--seed-offset',str(10000+batch*10),'--json',str(dest)]
            state['activeBatch']=batch
            state['updated']=now()
            save({**state,**summarize(samples),'samples':samples})
            with (OUT/('batch_%02d.log'%batch)).open('w',encoding='utf-8') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,check=True,timeout=7200)
            data=json.loads(dest.read_text(encoding='utf-8-sig'))
            for item in data['samples']:
                samples.append({**item,'game':item['game']+batch*20,'batch':batch})
            state['updated']=now()
            save({**state,**summarize(samples),'samples':samples})
        else:
            state['status']='complete'
        state['finished']=now()
        save({**state,**summarize(samples),'samples':samples})
        if state['status']=='complete':
            stats=summarize(samples)
            p=stats['score']
            elo=400*math.log10(p/(1-p)) if 0<p<1 else None
            text='# Delta版の長めの持ち時間での確認\n\n'
            text+='600局、1手500ms、Threads1、Hash64MB、NetworkDelay0、1組ずつ。新旧ともR5。低CPU優先度でfloodgate対局と並行。\n\n'
            text+='%d勝%d敗%d分、得点率%.2f%%。旧版とのElo差は%s。\n\n'%(stats['wins'],stats['losses'],stats['draws'],p*100,round(elo,1) if elo is not None else '算出不可')
            ci=stats['paired95']
            text+='先後ペアを単位とした95%%の概算区間: %.2f〜%.2f%%。\n\n'%(ci[0]*100,ci[1]*100)
            text+='1手50msの以前の比較とは別の実験。500msでの結果であり、floodgateの5分＋10秒や別エンジンへの強さを直接保証しない。ランダムな歩4手の序盤、seed10001〜10300。同じ序盤の偶然の重複は未除外。20局ごとにエンジンを起動し直す。\n'
            text+='並行するfloodgateの負荷は変動し得る。実測の持ち時間条件を明記して評価する。生結果はbaseline/delta_long_500ms_20261003/progress.jsonとbatchログ。\n'
            (ROOT/'REPORT_Delta長時間確認_20261003.md').write_text(text,encoding='utf-8')
    except Exception:
        state.update(status='failed',finished=now(),error=traceback.format_exc())
        save({**state,**summarize(samples),'samples':samples})
        raise
if __name__=='__main__':
    main()
