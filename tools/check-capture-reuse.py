from pathlib import Path
import json,subprocess,statistics,math,random,datetime,traceback
ROOT=Path(__file__).resolve().parent.parent
out=ROOT/'baseline/capture_reuse_state.json'
state=dict(status='running',started=datetime.datetime.now().isoformat(),feat=4325425)
def save():out.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
def run(args,path,timeout=1200):
 with (ROOT/path).open('w',encoding='utf-8') as f:
  subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,check=True,timeout=timeout)
try:
 assert json.loads((ROOT/'baseline/danger_experiment_state.json').read_text(encoding='utf-8'))['status']=='confirmation_complete'
 save()
 run(['cargo','test','--target-dir','target_capture_reuse'],'baseline/capture_reuse_tests.log')
 run(['cargo','build','--release','--bin','sailfish2','--target-dir','target_capture_reuse'],'baseline/capture_reuse_build.log')
 run(['node','--no-warnings','tools/compare-bench.mjs','baseline/sailfish2_time_candidate.exe','target_capture_reuse/release/sailfish2.exe|Feat=4325425','11','7'],'baseline/capture_reuse_bench.json',1800)
 result=json.loads((ROOT/'baseline/capture_reuse_bench.json').read_text(encoding='utf-8'))
 assert result['identical'] and result['stable'],'Fixed-depth outputs changed'
 ratios=[a['totalMs']/b['totalMs'] for a,b in zip(*result['runs'])]
 rng=random.Random(20261004)
 boot=sorted(math.exp(statistics.mean(math.log(rng.choice(ratios)) for _ in ratios))-1 for _ in range(10000))
 state.update(status='complete',finished=datetime.datetime.now().isoformat(),identical=True,depth=11,rounds=7,medianMs=result['medianMs'],speedupPercent=result['speedupPercent'],pairedRatios=ratios,pairedSpeed95Percent=[boot[250]*100,boot[9750]*100])
 save();print(json.dumps(state,ensure_ascii=False),flush=True)
except Exception:
 state.update(status='failed',error=traceback.format_exc());save();raise
