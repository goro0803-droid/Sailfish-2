from pathlib import Path
import sys,subprocess,json,hashlib,collections,datetime
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'data/mekitto_runtime'))
import cshogi,numpy as np
OUT=ROOT/'data/mekitto_public_20190212'
def mirror_key(sfen):
    board,side,hands,*_=sfen.split()
    ranks=[]
    for rank in board.split('/'):
        cells=[];i=0
        while i<len(rank):
            c=rank[i]
            if c.isdigit():cells.extend(['.']*int(c))
            elif c=='+':i+=1;cells.append('+'+rank[i])
            else:cells.append(c)
            i+=1
        assert len(cells)==9
        assert len(cells)==9
        out='';empty=0
        for cell in reversed(cells):
            if cell=='.':empty+=1
            else:
                if empty:out+=str(empty);empty=0
                out+=cell
        if empty:out+=str(empty)
        ranks.append(out)
    return '/'.join(ranks)+' '+side+' '+hands
def main():
    if (OUT/'sample_audit.json').exists():raise RuntimeError('Sample already exists')
    proc=subprocess.Popen(['C:/Program Files/7-Zip/7z.exe','x','-so',str(OUT/'shuffled_20190212.7z'),'shuffled_20190212.bin'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    counts=collections.Counter();seen=set();rows=[[],[]];scores=[];turns=collections.Counter();results=collections.Counter()
    board=cshogi.Board()
    try:
        raw=bytearray()
        while len(raw)<40000*40:
            part=proc.stdout.read(40000*40-len(raw))
            if not part:break
            raw.extend(part)
        assert len(raw)==40000*40,'Unexpected record stream length'
        data=np.frombuffer(raw,dtype=cshogi.PackedSfenValue)
        for r in data:
            counts['read']+=1
            try:
                board.set_psfen(np.array(r['sfen'],copy=True))
                move=board.move_from_move16(cshogi.move16_from_psv(int(r['move'])))
                if not board.is_legal(move):counts['illegal_move']+=1;continue
                score=int(r['score']);res=int(r['game_result']);ply=int(r['gamePly'])
                if res not in [-1,0,1]:counts['invalid_result']+=1;continue
                if abs(score)>2000 or ply<16 or board.is_check():counts['filtered']+=1;continue
                sfen=board.sfen()
                key=' '.join(sfen.split()[:3])
                canonical=min(key,mirror_key(sfen))
                if canonical in seen:counts['duplicate']+=1;continue
                seen.add(canonical)
                split=int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8],'little')%20==0
                rows[int(split)].append('P\t'+sfen+'\t'+str(score)+'\t'+str((res+1)/2)+'\t'+cshogi.move_to_usi(move)+'\t'+str(ply))
                scores.append(score);turns[str(board.turn)]+=1;results[str(res)]+=1;counts['accepted']+=1
            except Exception as e:
                counts['decode_error']+=1
                if counts['decode_error']<=3:print('Decode error:',repr(e),flush=True)
        if counts['illegal_move'] or counts['decode_error'] or counts['invalid_result']:
            raise RuntimeError('Format validation failed: '+str(dict(counts)))
        for name,lines in zip(['sample_train.tsv','sample_validation.tsv'],rows):
            (OUT/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        audit=dict(source='public_teacher_by_mekitto.7z',inner='shuffled_20190212.bin',format='PackedSfenValue 40 bytes',sample='first 40000 records of pre-shuffled file; bounded stream, full archive CRC not checked',counts=dict(counts),train=len(rows[0]),validation=len(rows[1]),turns=dict(turns),results=dict(results),scoreMin=min(scores),scoreMax=max(scores),created=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),split='canonical position including horizontal mirror grouped by SHA256 modulo20; no source game identifiers, game-level disjointness cannot be established')
        (OUT/'sample_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(audit,ensure_ascii=False),flush=True)
    finally:
        proc.kill()
        proc.communicate()
if __name__=='__main__':main()
