from pathlib import Path
import runpy,hashlib
R=Path(__file__).resolve().parent.parent
mirror=runpy.run_path(str(R/'tools/sample-mekitto.py'))['mirror_key']
rows=(R/'data/mekitto_public_20190212/sample_train.tsv').read_text(encoding='utf-8').splitlines()[:1000]
def canon(s):return min(' '.join(s.split()[:3]),mirror(s))
for line in rows:
 s=line.split('\t')[1];key=' '.join(s.split()[:3]);assert mirror(mirror(s))==key;assert canon(s)==canon(mirror(s))
 assert hashlib.sha256(canon(s).encode()).digest()==hashlib.sha256(canon(mirror(s)).encode()).digest()
s='4k4/9/3+p5/9/1+B7/9/9/9/4K4 w 2R3p 19'
assert mirror(mirror(s))==' '.join(s.split()[:3]);assert canon(s)==canon(mirror(s))
print('1001 mirror/canonical/grouping checks passed; promotion and hand fixture included')
