import pandas as pd, numpy as np
df=pd.read_csv(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)),"epi_gen.csv")); df=df[(df.pm==1.0)&(df.age>=1)].copy()
RL={"steel/KES@37":2,"rare_earths/VAL@136":3,"copper/ORI@132":4,"natural_gas/VAL@153":2,"steel/HAL@158":2,"lithium/KES@215":5}
def rule_p(ep,age,k,alpha=0.5,fizz=0.1,maxL=8):
    others=[v for e,v in RL.items() if e!=ep]
    Ls=np.arange(1,maxL+1); w=np.array([sum(o==L for o in others) for L in Ls],float)+alpha
    w[Ls<age]=0  # run on at origin => onset after origin => L>=age
    w=w/w.sum()*(1-fizz)
    # onset horizon from origin: first+L - origin = L-age+1
    hz=Ls-age+1; return float(w[hz==k].sum()) if (hz==k).any() else 0.0, w, hz
out=[]
for _,r in df.iterrows():
    k=int(r.actual-r.origin); pr,_,_=rule_p(r.ep,int(r.age),k)
    pg=r.p_at_actual
    out.append(dict(ep=r.ep,origin=r.origin,age=r.age,k=k,gen=pg,rule=round(pr,3),mix=round(0.5*pg+0.5*pr,3),
                    pit=round(r.P_by_actual-0.5*pg,3),in80=r.in80))
o=pd.DataFrame(out); print(o.to_string(index=False))
ll=lambda p:np.log(np.maximum(p,1e-3))
print("mean log score  gen %.2f  rule(LOO) %.2f  mix %.2f"%(ll(o.gen).mean(),ll(o.rule).mean(),ll(o.mix).mean()))
print("mean p at actual gen %.3f rule %.3f mix %.3f"%(o.gen.mean(),o.rule.mean(),o.mix.mean()))
print("80%% coverage %d/%d; mean PIT %.2f (0.5 = centred; >0.5 = onset later than generator says)"%(o.in80.sum(),len(o),o.pit.mean()))
# origin 221 implied distributions (all 6 episodes, no LOO)
RL["_none"]=None
for name,age,gen in [("VAL cobalt",3,[0.286,0.566,0.119,0.028,0,0]),("KES rare_earths",1,[0.028,0.422,0.502,0.046,0.001,0])]:
    _,w,hz=rule_p("_none",age,1)
    d={221+int(h):round(x,3) for h,x in zip(hz,w) if x>0.001 and h>=1}
    print(name,"rule:",d,"(fizzle 0.10)  gen:",{222+i:g for i,g in enumerate(gen) if g})
