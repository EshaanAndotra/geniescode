import pandas as pd, numpy as np
import os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","Data_new")+"/"
pr=pd.read_csv(D+"prices.csv"); sh=pd.read_csv(D+"supply_history.csv"); sup=pd.read_csv(D+"suppliers.csv")
pr["prem"]=pr.futures_4w/pr.spot_price-1
sh=sh.merge(sup[["material","country","normal_capacity_units","lane"]],on=["material","country"])
sh["cr"]=sh.capacity_units/sh.normal_capacity_units
df=pr.merge(sh[["week","material","country","cr","lane"]],on=["week","material","country"],how="left")
rows=[]
for (m,c),g in df.groupby(["material","country"]):
    g=g.sort_values("week"); w=g.week.values; p=g.prem.values; cr=g.cr.values
    above=p>0.06
    # runs
    i=0
    while i<len(w):
        if above[i]:
            j=i
            while j+1<len(w) and above[j+1]: j+=1
            # next shock onset (cr<0.7) after run start
            later=np.where((cr<0.7)&(w>=w[i]))[0]
            on=w[later[0]] if len(later) else None
            # is there a shock during/at run start already?
            inshock=bool(cr[i]<0.7) if np.isfinite(cr[i]) else None
            rows.append(dict(mat=m,cty=c,run_start=w[i],run_end=w[j],len=j-i+1,peak=p[i:j+1].max().round(3),
                             prem_end=p[j].round(3),next_cap_onset=on,gap_from_end=(on-w[j]) if on else None,in_shock_at_start=inshock))
            i=j+1
        else: i+=1
ep=pd.DataFrame(rows); pd.set_option("display.width",200)
print(ep.to_string())
# single-supplier shocks: cap<0.7 runs
srows=[]
for (m,c),g in df.groupby(["material","country"]):
    g=g.sort_values("week"); w=g.week.values; cr=g.cr.values; p=g.prem.values
    lo=np.nan_to_num(cr,nan=1)<0.7
    i=0
    while i<len(w):
        if lo[i]:
            j=i
            while j+1<len(w) and lo[j+1]: j+=1
            srows.append(dict(mat=m,cty=c,on=w[i],off=w[j],len=j-i+1,mincap=cr[i:j+1].min().round(2),
                              prem_m4=p[max(0,i-4):i].round(3).tolist()))
            i=j+1
        else: i+=1
s=pd.DataFrame(srows).sort_values("on"); print(s.to_string())
# crossings count (weeks where prem crosses above 0.06)
cross=((df.sort_values(["material","country","week"]).groupby(["material","country"]).prem.apply(lambda x:((x>0.06)&(x.shift(1)<=0.06)).sum())))
print("crossings total",cross.sum(), " weeks above 0.06:",(df.prem>0.06).sum())
