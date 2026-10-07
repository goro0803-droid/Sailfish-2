from pathlib import Path
import json,subprocess,threading,queue,time,re
ROOT=Path(__file__).resolve().parent.parent
data=json.loads((ROOT/'baseline/floodgate_sailfish2_analysis.json').read_text(encoding='utf-8'))
game=next(r for r in data['games'] if r['start'].endswith('03:30:00'))
rows=[m for m in game['moves'] if m['side']==game['side']][:10]
p=subprocess.Popen([str(ROOT/'target_time/release/sailfish2.exe')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW)
q=queue.Queue()
def read():
    for line in p.stdout:q.put(line.strip())
threading.Thread(target=read,daemon=True).start()
def send(s):p.stdin.write(s+'\n');p.stdin.flush()
def wait(prefix,seconds=40):
    deadline=time.monotonic()+seconds;seen=[]
    while True:
        l=q.get(timeout=max(.01,deadline-time.monotonic()));seen.append(l)
        if l.startswith(prefix):return l,seen
try:
    send('usi');wait('usiok')
    send('setoption name Threads value 4')
    send('setoption name USI_Hash value 64')
    send('isready');wait('readyok')
    send('usinewgame')
    bank=300000;results=[]
    for row in rows:
        prefix=[m['usi'] for m in game['moves'] if m['ply']<row['ply']]
        send('position startpos moves '+' '.join(prefix))
        own=game['side']
        clocks=[300000,300000];clocks[own]=int(bank)
        t=time.monotonic()
        send('go btime %d wtime %d binc 10000 winc 10000'%(clocks[0],clocks[1]))
        answer,lines=wait('bestmove')
        ms=(time.monotonic()-t)*1000
        assert ms<=15000,'Opening cap exceeded'
        assert not any('illegal' in l.lower() for l in lines)
        bank=bank-ms+10000
        result=dict(ownMove=row['ownN'],sourcePly=row['ply'],historicalMs=row['time']*1000,newMs=round(ms),remainingMs=round(bank),best=answer,info=next((l for l in reversed(lines) if l.startswith('info depth')),''))
        results.append(result)
        (ROOT/'baseline/time_opening_trace.json').write_text(json.dumps(dict(source=game['file'],threads=4,initialMs=300000,incrementMs=10000,rows=results),ensure_ascii=False,indent=2),encoding='utf-8')
        print('move',row['ownN'],'new ms',round(ms),'bank ms',round(bank),flush=True)
    assert bank>=250000,'Opening reserve not maintained'
    print('Opening reserve verified',round(bank),flush=True)
finally:
    send('quit')
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill()
