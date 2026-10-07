from pathlib import Path
import sys,json,re,statistics,collections
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'data/mekitto_runtime'))
import cshogi
def analyze(path):
    text=path.read_text(encoding='utf-8')
    names={0:re.search(r'^N\+(.*)$',text,re.M).group(1),1:re.search(r'^N-(.*)$',text,re.M).group(1)}
    side=next(s for s,n in names.items() if n=='sailfish2')
    board=cshogi.Board()
    moves=[];clocks=[300,300];ownN=0
    for line in text.splitlines():
        if re.fullmatch(r'[+-]\d{4}[A-Z]{2}',line):
            mover=0 if line[0]=='+' else 1
            assert board.turn==mover
            move=board.move_from_csa(line[1:])
            assert board.is_legal(move),(path.name,line)
            if mover==side:ownN+=1
            moves.append(dict(ply=len(moves)+1,side=mover,ownN=ownN,usi=cshogi.move_to_usi(move),beforeSfen=board.sfen(),clockBefore=clocks[mover],time=None,scoreBlack=None))
            board.push(move)
        elif re.fullmatch(r'T\d+',line):
            assert moves and moves[-1]['time'] is None
            t=int(line[1:]);row=moves[-1];row['time']=t
            clocks[row['side']]-=t
            row['clockBeforeIncrement']=clocks[row['side']]
            clocks[row['side']]+=10
            row['clockAfter']=clocks[row['side']]
        elif re.match(r"'\*\* -?\d+",line):
            row=moves[-1];row['scoreBlack']=int(line.split()[1])
    assert all(m['time'] is not None for m in moves)
    own=[m for m in moves if m['side']==side]
    for m in moves:
        m['scoreOwn']=None if m['scoreBlack'] is None else m['scoreBlack']*(1 if side==0 else -1)
    summary=re.search(r"^'summary:(.*)$",text,re.M).group(1)
    outcome='win' if 'sailfish2 win' in summary else 'loss' if 'sailfish2 lose' in summary else 'other'
    rates=re.findall(r"^'(black|white)_rate:.*:(-?\d+)$",text,re.M)
    rates={0 if s=='black' else 1:int(r) for s,r in rates}
    start=re.search(r'^\$START_TIME:(.*)$',text,re.M).group(1)
    termination=re.search(r'^%(.+)$',text,re.M).group(1)
    def checkpoint(n):
        if len(own)<n:return None
        row=own[n-1]
        return dict(moveNumber=n,ply=row['ply'],spent=sum(m['time'] for m in own[:n]),remaining=row['clockAfter'],scoreOwn=row['scoreOwn'])
    ownScores=[m for m in own if m['scoreOwn'] is not None]
    crossings={str(k):next((dict(ply=m['ply'],ownN=m['ownN'],score=m['scoreOwn'],remaining=m['clockAfter']) for m in ownScores if m['scoreOwn']<=k),None) for k in [-500,-1000]}
    swings=[]
    for a,b in zip(ownScores,ownScores[1:]):
        drop=a['scoreOwn']-b['scoreOwn']
        if drop>=300:
            swings.append(dict(drop=drop,fromPly=a['ply'],toPly=b['ply'],before=a['scoreOwn'],after=b['scoreOwn'],beforePosition=a['beforeSfen'],played=a['usi'],nextPosition=b['beforeSfen']))
    return dict(file=path.name,url='https://wdoor.c.u-tokyo.ac.jp/shogi/x/2026/'+re.search(r'\d{4}(\d{2})(\d{2})\d{6}\.csa$',path.name).expand(r'\1/\2/')+path.name,start=start,side=side,opponent=names[side^1],opponentRate=rates.get(side^1),outcome=outcome,termination=termination,plies=len(moves),ownMoves=len(own),checkpoints={str(n):checkpoint(n) for n in [5,10,15,20,30,40]},first10Mean=statistics.mean(m['time'] for m in own[:10]),maxThink=max(m['time'] for m in own),remainingMin=min(m['clockBeforeIncrement'] for m in own),finalRemaining=own[-1]['clockAfter'],last10Mean=statistics.mean(m['time'] for m in own[-10:]),below60At=next((m['ownN'] for m in own if m['clockAfter']<60),None),crossings=crossings,swings=sorted(swings,key=lambda x:-x['drop'])[:3],moves=moves)
def main():
    # 使い方: python analyze-sailfish-floodgate.py [棋譜フォルダ] [出力JSON] [主集計の局数]（省略時は旧IDの15局）
    src=sys.argv[1] if len(sys.argv)>1 else 'data/floodgate_sailfish2_20261003'
    dst=sys.argv[2] if len(sys.argv)>2 else 'baseline/floodgate_sailfish2_analysis.json'
    nmain=int(sys.argv[3]) if len(sys.argv)>3 else 15
    rows=sorted([analyze(p) for p in (ROOT/src).glob('*.csa')],key=lambda r:r['start'])
    main=rows[:nmain];extra=rows[nmain:]
    summary=dict(mainGames=len(main),outcomes=dict(collections.Counter(r['outcome'] for r in main)),extraGames=len(extra),first10SpentMedian=statistics.median(r['checkpoints']['10']['spent'] for r in main),remainingAfter10Median=statistics.median(r['checkpoints']['10']['remaining'] for r in main),remainingAfter20Median=statistics.median(r['checkpoints']['20']['remaining'] for r in main),first10Mean=statistics.mean(m['time'] for r in main for m in r['moves'] if m['side']==r['side'] and m['ownN']<=10),last10Mean=statistics.mean(r['last10Mean'] for r in main),below60Games=sum(r['below60At'] is not None for r in main),earlyBelow60Games=sum(r['below60At'] is not None and r['below60At']<=20 for r in main),lossCrossing500WithAbove60=sum(r['outcome']=='loss' and r['crossings']['-500'] is not None and r['crossings']['-500']['remaining']>60 for r in main))
    out=dict(summary=summary,games=main,extraGames=extra,clockAssumption='300 seconds + 10 seconds after each own move; CSA integer T seconds, reconstructed clocks approximate',scoreAssumption='CSA evaluation comments are black-side perspective; own evaluation compared only across sailfish2 own moves, not opponent scale')
    (ROOT/dst).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    for r in rows:print(json.dumps({k:r[k] for k in ['start','opponent','opponentRate','outcome','first10Mean','maxThink','remainingMin','last10Mean','below60At','crossings']},ensure_ascii=False))
if __name__=='__main__':main()
