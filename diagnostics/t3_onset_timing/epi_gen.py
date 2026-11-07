import sys,os,time,warnings
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")
sys.path.insert(0,os.path.join(ROOT,"Code Base","pipelining","Model 1","iter4"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
DATA=os.path.join(ROOT,"Data_new")
# (material, country, first premium>0.06 week, onset week)
EPI=[("steel","KES",35,37),("rare_earths","VAL",133,136),("copper","ORI",128,132),
     ("natural_gas","VAL",151,153),("steel","HAL",156,158),("lithium","KES",210,215),
     ("rare_earths","VAL",None,78),("aluminum","HAL",None,195)]   # last two: no premium warning
def init():
    global m1,P
    import model1 as _m; m1=_m
    P=m1.Panels(m1.load_data(DATA))
def one(o):
    M=m1.DisruptionModel().fit(P,origin=o); src=M._source_probs(13)
    return o,src["q_sup"].copy(),list(src["sup_state"])
if __name__=="__main__":
    jobs=set()
    for m,c,f,on in EPI:
        lo=(f if f else on)-2
        jobs|=set(range(lo,on))
    jobs|={221}
    t=time.time()
    with Pool(4,initializer=init) as pool: res=dict((o,(q,s)) for o,q,s in pool.map(one,sorted(jobs)))
    print("fits",len(jobs),"in",round(time.time()-t),"s")
    import model1; sup=pd.read_csv(DATA+"/suppliers.csv")
    rows=[]
    for m,c,f,on in EPI:
        u=int(np.where((sup.material==m)&(sup.country==c))[0][0])
        lo=(f if f else on)-2
        for o in range(lo,on):
            q,st=res[o]; qu=q[u]
            for pm in (1.0,1.5):
                qq=np.clip(qu*pm,0,1); qq=qq/qq.sum() if qq.sum()>1 else qq
                k=on-o  # actual onset horizon
                cdf=np.cumsum(qq)
                pw=qq.sum()
                cond=qq/pw if pw>0 else qq
                mean=(np.arange(1,14)*cond).sum()+o if pw>0 else np.nan
                ccdf=np.cumsum(cond)
                lo10=o+1+int(np.searchsorted(ccdf,0.1)); hi90=o+1+int(np.searchsorted(ccdf,0.9))
                rows.append(dict(ep=f"{m}/{c}@{on}",origin=o,age=(o-f+1) if f else None,state=st[u],pm=pm,
                    P_in13=round(pw,3),p_at_actual=round(qq[k-1],3),P_by_actual=round(cdf[k-1],3),
                    P_within1=round(qq[max(0,k-2):k+1].sum(),3),cond_mean=round(mean,2),cond80=f"{lo10}-{hi90}",actual=on,
                    in80=(lo10<=on<=hi90)))
    df=pd.DataFrame(rows); pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"epi_gen.csv"),index=False)
    print(df[df.pm==1.0].drop(columns="pm").to_string(index=False))
    # origin 221 dump
    q,st=res[221]
    for m,c in [("cobalt","VAL"),("rare_earths","KES")]:
        u=int(np.where((sup.material==m)&(sup.country==c))[0][0]); print(m,c,np.round(q[u][:6],3))
