import re,glob,sys,os,collections,statistics as st,json,gzip
S=os.environ.get("AB_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "outputs", "experiments"))
cats={}
for line in open("/data/pengyh/workspace/FreeAskAgent_R2R/integrations/v3/eval_sets/val_unseen_40.txt"):
    line=line.split("#")[0].strip()
    if line: i,c=line.split(","); cats[i]=c
geo={}
_paths=["/data/pengyh/workspace/habitat/data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz","/data/pengyh/workspace/AwareVLN/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"]
d=json.load(gzip.open(next(p for p in _paths if os.path.exists(p))))
for e in d["episodes"]: geo[str(e["episode_id"])]=e["info"]["geodesic_distance"]
def load(cfg):
    rows={}
    for f in glob.glob(f"{S}/ab/{cfg}/rank_*.log"):
        t=open(f,errors="replace").read()
        # split per episode
        for m in re.finditer(r"^episode=(\d+) instruction=.*?(?=^episode=\d+ instruction=|\Z)",t,re.M|re.S):
            ep=m.group(1); blk=m.group(0)
            r=re.search(r"id=%s steps=(\d+) success=([\d.]+) spl=([\d.]+) dtg=([\d.]+)"%ep,blk)
            if not r: continue
            steps=[int(x) for x in re.findall(r"^ep=\d+ s=\d+ (\d+)ms",blk,re.M)]
            wp=sum(1 for x in re.findall(r"sel=(\d+) cap",blk) if int(x)>50)
            scene=sum(1 for x in re.findall(r"cap=(\d+) pre",blk) if int(x)>0)
            subs=len(re.findall(r"^  \[(\d+)\]",blk,re.M)); sg=[int(g) for g in re.findall(r"sg=\S+->(\d+)",blk)]
            sp=re.findall(r"sp=(\w+):(\w+) d=([\d.-]+) age=(\d+)",blk)
            rows[ep]=dict(steps=int(r[1]),succ=float(r[2]),spl=float(r[3]),dtg=float(r[4]),step_ms=st.mean(steps) if steps else 0,
                          wp=wp,scene=scene,sgfrac=(max(sg)/subs if sg and subs else 0),reached_final=(bool(sg) and subs and max(sg)==subs),
                          stopped=("act=STOP" in blk),sp_steps=len(sp),commits=sum(1 for k,s,d,a in sp if a=="0"),
                          frontier=sum(1 for k,s,d,a in sp if k=="frontier"),cat=cats.get(ep,"?"),geo=geo.get(ep,0))
    return rows
cfgs=sys.argv[1:] or ["spatial_off","spatial_on"]
loaded={c:load(c) for c in cfgs}
common=sorted(set.intersection(*[set(v) for v in loaded.values()]),key=int)
print("episodes:",{c:len(v) for c,v in loaded.items()},"common",len(common))
off=loaded[cfgs[0]]; on=loaded[cfgs[-1]]
def agg(rows,eps):
    r=[rows[e] for e in eps]
    if not r: return None
    prog=[(x['geo']-x['dtg'])/x['geo'] for x in r if x['geo']>0]
    return dict(n=len(r),SR=st.mean(x['succ'] for x in r),SPL=st.mean(x['spl'] for x in r),dtg=st.mean(x['dtg'] for x in r),
                prog=st.mean(prog),final=st.mean(1.0 if x['reached_final'] else 0.0 for x in r),stopped=st.mean(1.0 if x['stopped'] else 0.0 for x in r),
                wp=st.mean(x['wp'] for x in r),scene=st.mean(x['scene'] for x in r),step_ms=st.mean(x['step_ms'] for x in r),steps=st.mean(x['steps'] for x in r))
def show(title,eps):
    print(f"\n### {title} (n={len(eps)})")
    print("| config | SR | SPL | dtg m | progress | reached final | stopped | wp calls | scene calls | step ms | steps |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name,rows in loaded.items():
        a=agg(rows,eps)
        if a: print(f"| {name} | {a['SR']:.2f} | {a['SPL']:.2f} | {a['dtg']:.1f} | {a['prog']:.2f} | {a['final']:.2f} | {a['stopped']:.2f} | {a['wp']:.0f} | {a['scene']:.0f} | {a['step_ms']:.0f} | {a['steps']:.0f} |")
show("ALL",common)
for c in ("doorway","walk","stairs","abstract_stop"):
    show(c,[e for e in common if cats.get(e)==c])
print("\nper-episode (on vs off): ep cat | succ dtg sgfrac | succ dtg sgfrac | commits frontier")
for e in common:
    a,b=on[e],off[e]
    flag="  <-- on better" if a['dtg']<b['dtg']-1 else ("  <-- off better" if b['dtg']<a['dtg']-1 else "")
    print(f"{e:>5} {a['cat']:13s}| on: {a['succ']:.0f} {a['dtg']:5.1f} {a['sgfrac']:.2f} | off: {b['succ']:.0f} {b['dtg']:5.1f} {b['sgfrac']:.2f} | commits={a['commits']:2d} frontier={a['frontier']:2d}{flag}")
