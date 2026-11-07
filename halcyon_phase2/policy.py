"""Finite-window purchasing policy and transparent local simulator.

The organizer's simulator implementation was not supplied. These functions
implement the stated mechanics, with explicit conventions documented below.
Only plan_policy is predictive; simulate_policy deliberately uses realized data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

COUNTRIES = ["KES", "VAL", "SUN", "ORI", "HAL", "AUR"]


def _policy_frame(policy):
    p = policy.copy()
    if "week" not in p:
        p["week"] = p.id.str.extract(r"^w(\d+)_").astype(int)
        p["material"] = p.id.str.replace(r"^w\d+_", "", regex=True)
    return p


def _initial_stock(demand, material, start, end, origin=None, convention="window_mean"):
    d = demand[demand.material.eq(material)]
    if convention == "window_mean":
        rate = d.loc[d.week.between(start, end), "demand_units"].mean()
    elif convention == "historical_mean":
        rate = d.loc[d.week.le(origin or start - 1), "demand_units"].mean()
    elif convention == "first_week":
        rate = d.loc[d.week.eq(start), "demand_units"].iloc[0]
    else:
        raise ValueError(f"Unknown initial-stock convention: {convention}")
    return 5.0 * float(rate)


def naive_policy(data, origin, start, end):
    """Sample analogue: each week's known demand, cheapest historical supplier."""
    hist = data["supply_history"].query("week <= @origin")
    demand = data["demand"]
    rows = []
    template = _policy_frame(data["sample_submission"])
    for mat, supp in data["suppliers"].groupby("material"):
        costs = hist[hist.material.eq(mat)].groupby("country").landed_cost.mean()
        country = costs.idxmin()
        if origin == 208 and start >= 209:
            rr = template[template.material.eq(mat)].iloc[0]
            country = max(COUNTRIES, key=lambda c: rr[f"share_{c}"])
        for week in range(start, end + 1):
            quantity = float(demand.loc[demand.material.eq(mat) & demand.week.eq(week), "demand_units"].iloc[0])
            rows.append(dict(week=week, material=mat, order_units=quantity,
                             **{f"share_{c}": float(c == country) for c in COUNTRIES}))
    return pd.DataFrame(rows)


def tune_policy(input_directory, output_path):
    """Small, fixed development grid; origin 182 is reserved for evaluation."""
    import json
    from pathlib import Path
    folder = Path(input_directory)
    data = {x:pd.read_csv(folder/f"{x}.csv") for x in
            ["supply_history","demand","suppliers","outcomes","sample_submission","tariffs"]}
    results = []
    for safety,maxorder in [(.55,10),(.55,1.6),(1.0,1.6),(1.5,1.6)]:
        for origin in [104,130,156]:
            for gap in [0,13]:
                start,end = origin+gap+1,origin+gap+13
                a = simulate_policy(data,plan_policy(data,origin,start,end,safety_weeks=safety,max_order_weeks=maxorder),start,end,origin)
                b = simulate_policy(data,naive_policy(data,origin,start,end),start,end,origin)
                row = dict(safety=safety,maxorder=maxorder,origin=origin,gap=gap,
                           cost_saving=1-a["cost"]/b["cost"],fill=a["fill_rate"],
                           naive_fill=b["fill_rate"],shock_fill=a["shock_fill_rate"],
                           naive_shock_fill=b["shock_fill_rate"],div=a["diversification"])
                results.append(row)
                print(json.dumps(row),flush=True)
    Path(output_path).write_text(json.dumps(results,indent=2))
    return results


def simulate_policy(data, policy, start, end, origin=None,
                    initial_stock="window_mean", holding_on="ending"):
    """Evaluate realized purchases, cancellation, arrival and inventory costs.

    Conventions: lead L ordered at t arrives at t+L; initial stock equals five
    times mean window demand; cost reference equals mean window landed cost
    across three suppliers; holding charged on ending on-hand inventory;
    shortage is lost sales; salvage includes all accepted in-transit units.
    Aggregate fill is both unit-weighted and material-averaged.
    """
    p = _policy_frame(policy)
    hist = data["supply_history"].set_index(["week", "material", "country"])
    demand = data["demand"].set_index(["week", "material"]).demand_units
    outcomes = data["outcomes"]
    details, weekly = [], []
    for mat, pp in p[p.week.between(start, end)].groupby("material"):
        pp = pp.set_index("week")
        ref = data["supply_history"].loc[
            data["supply_history"].material.eq(mat) & data["supply_history"].week.between(start, end),
            "landed_cost"].mean()
        if not np.isfinite(ref):
            raise ValueError("Realized supply data unavailable for requested simulation window")
        inv = _initial_stock(data["demand"], mat, start, end, origin, initial_stock)
        arrivals = {}
        purchase = holding = shortage_cost = used = needed = shock_used = shock_needed = accepted_total = cancelled_total = 0.0
        shock_weeks = set()
        for w in outcomes.loc[outcomes.material.eq(mat) & outcomes.disrupted.eq(1), "week"]:
            shock_weeks.update(range(int(w), int(w) + 5))
        div, order_div, order_weights = [], [], []
        for week in range(start, end + 1):
            r = pp.loc[week]
            inv += arrivals.pop(week, 0.0)
            for country in COUNTRIES:
                requested = float(r.order_units) * float(r[f"share_{country}"])
                if requested <= 1e-10:
                    continue
                supply = hist.loc[(week, mat, country)]
                accepted = min(requested, float(supply.capacity_units))
                purchase += accepted * float(supply.landed_cost)
                accepted_total += accepted
                cancelled_total += requested - accepted
                arrival = week + int(supply.lead_time_weeks)
                if arrival == week:
                    inv += accepted
                else:
                    arrivals[arrival] = arrivals.get(arrival, 0.0) + accepted
            d = float(demand.loc[(week, mat)])
            opening = inv
            filled = min(inv, d)
            inv -= filled
            missed = d - filled
            used += filled
            needed += d
            shortage_cost += 5.0 * ref * missed
            holding += 0.02 * ref * (inv if holding_on == "ending" else opening)
            if week in shock_weeks:
                shock_used += filled
                shock_needed += d
            shares = np.array([float(r[f"share_{c}"]) for c in COUNTRIES])
            diversification = float(np.clip((1.0 - (shares**2).sum()) / (1.0 - 1/3), 0, 1))
            div.append(diversification)
            order_div.append(diversification * float(r.order_units))
            order_weights.append(float(r.order_units))
            weekly.append(dict(week=week, material=mat, demand=d, filled=filled,
                               ending_inventory=inv, shortage=missed, shock=week in shock_weeks))
        transit = sum(arrivals.values())
        salvage = 0.7 * ref * (inv + transit)
        details.append(dict(material=mat, cost=purchase+holding+shortage_cost-salvage,
                            purchase_cost=purchase, holding_cost=holding, shortage_cost=shortage_cost,
                            salvage=salvage, demand=needed, filled=used, fill_rate=used/needed,
                            shock_demand=shock_needed, shock_filled=shock_used,
                            shock_fill_rate=shock_used/shock_needed if shock_needed else None,
                            diversification=float(np.mean(div)),
                            order_weighted_diversification=sum(order_div)/max(sum(order_weights), 1e-9),
                            ending_inventory=inv, in_transit=transit,
                            accepted_units=accepted_total, cancelled_units=cancelled_total))
    dd = pd.DataFrame(details)
    total = {k:float(dd[k].sum()) for k in ["cost", "purchase_cost", "holding_cost", "shortage_cost", "salvage", "demand", "filled", "shock_demand", "shock_filled"]}
    total.update(fill_rate=total["filled"]/total["demand"],
                 mean_material_fill_rate=float(dd.fill_rate.mean()),
                 shock_fill_rate=total["shock_filled"]/total["shock_demand"] if total["shock_demand"] else None,
                 mean_material_shock_fill_rate=float(dd.shock_fill_rate.dropna().mean()) if dd.shock_fill_rate.notna().any() else None,
                 diversification=float(dd.diversification.mean()),
                 by_material=details, weekly=weekly)
    return total


def _scenarios(history, suppliers, origin, start, horizon, count=12):
    """Empirical contiguous scenarios preserve correlated lane/country shocks."""
    countries = list(suppliers.country)
    basecap = suppliers.normal_capacity_units.to_numpy(float)
    baselead = suppliers.base_lead_weeks.to_numpy(int)
    h = history[history.week.le(origin)]
    latest = h.sort_values("week").groupby("country").tail(1).set_index("country").reindex(countries)
    rec = h[h.week.gt(origin-8)].groupby("country").landed_cost.median().reindex(countries).to_numpy()
    cur_cost = 0.7*latest.landed_cost.to_numpy()+0.3*rec
    starts = np.unique(np.round(np.linspace(1, max(1, origin-horizon+1), count)).astype(int))
    cap, lead, cost, weights = [], [], [], []
    # A normal-state scenario prevents sparse historical crises dominating.
    cap.append(np.tile(basecap*.97, (horizon, 1)))
    lead.append(np.tile(baselead, (horizon, 1)))
    cost.append(np.tile(cur_cost, (horizon, 1)))
    weights.append(.35)
    for first in starts:
        block = h[h.week.between(first, first+horizon-1)]
        cc = block.pivot(index="week", columns="country", values="capacity_units").reindex(index=range(first,first+horizon), columns=countries)
        ll = block.pivot(index="week", columns="country", values="lead_time_weeks").reindex(index=range(first,first+horizon), columns=countries)
        if cc.isna().any().any() or ll.isna().any().any():
            continue
        cap.append(cc.to_numpy(float)); lead.append(ll.to_numpy(int))
        cost.append(np.tile(cur_cost, (horizon, 1))); weights.append(.50/len(starts))
    # Continue current constraints initially and decay by order horizon; no
    # assumption that the entire future quarter matches the last observation.
    cc, ll = [], []
    for j in range(horizon):
        persistence = np.exp(-(start-origin-1+j)/4)
        cc.append(basecap*.97 + persistence*(latest.capacity_units.to_numpy()-basecap*.97))
        ll.append(np.maximum(baselead, np.ceil(baselead+persistence*(latest.lead_time_weeks.to_numpy()-baselead))).astype(int))
    cap.append(np.array(cc)); lead.append(np.array(ll)); cost.append(np.tile(cur_cost,(horizon,1))); weights.append(.15)
    weights = np.asarray(weights); weights /= weights.sum()
    return np.array(cap), np.array(lead), np.array(cost), weights, cur_cost


def _optimize_orders(demand, initial, cap, lead, costs, weights, reference,
                     normal_capacity, min_share=.23, max_share=.47,
                     safety_weeks=1.5, risk=None, max_order_weeks=1.6):
    """Scenario LP in demand-normalized units; fixed-point capacity response."""
    S, T, C = cap.shape
    scale = float(np.mean(demand))
    d = demand/scale; init = initial/scale
    caps = cap/scale
    normal = normal_capacity/scale
    nc = T*C
    ni = S*T
    n = nc+2*ni
    # Start from ordinary replenishment quantities to estimate cap cancellation.
    q = np.tile(d.mean()/C, (T,C))
    q[-int(np.min(lead)):,:] = 0
    a_ub = lil_matrix((T*C*2+T, n)); b_ub = np.zeros(T*C*2+T)
    k = 0
    for t in range(T):
        for c in range(C):
            for c2 in range(C):
                a_ub[k,t*C+c2] = min_share
                a_ub[k+1,t*C+c2] = -max_share
            a_ub[k,t*C+c] -= 1
            a_ub[k+1,t*C+c] += 1
            k += 2
        a_ub[T*C*2+t,t*C:(t+1)*C] = 1
        b_ub[T*C*2+t] = max_order_weeks
    bounds = [(0, float(.95*normal[c])) for t in range(T) for c in range(C)]
    bounds += [(0,None)]*ni
    bounds += [(0,float(d[t])) for s in range(S) for t in range(T)]
    for iteration in range(3):
        fill = np.minimum(1, caps/np.maximum(q[None,:,:], .15))
        # Retain a little cancellation conservatism as LP quantities change.
        if iteration:
            fill = .5*fill+.5*prev_fill
        prev_fill = fill
        objective = np.zeros(n)
        A = lil_matrix((ni,n)); b = np.zeros(ni)
        for s in range(S):
            for t in range(T):
                row = s*T+t; inv_index = nc+row
                A[row,inv_index] = 1
                if t: A[row,inv_index-1] = -1
                A[row,nc+ni+row] = -1
                b[row] = -d[t]+(init if t==0 else 0)
                objective[inv_index] = weights[s]*.02*reference
                # Penalize shortages above stated cost slightly to prefer
                # resilience where financial impact is nearly tied.
                objective[nc+ni+row] = weights[s]*6.0*reference
                if t==T-1: objective[inv_index] -= weights[s]*.70*reference
                for c in range(C):
                    index = t*C+c
                    objective[index] += weights[s]*fill[s,t,c]*costs[s,t,c]
                    arrival = t+lead[s,t,c]
                    if arrival<T:
                        A[s*T+arrival,index] -= fill[s,t,c]
                    else:
                        objective[index] -= weights[s]*fill[s,t,c]*.70*reference
        # Reserve a modest buffer in normal conditions, released toward the end.
        # A linear lower inventory bound on the first (normal) scenario remains
        # feasible because starting stock covers the first several weeks.
        bu = list(bounds)
        for t in range(T):
            reserve = safety_weeks*max(0, min(1,(T-1-t)/3))
            if risk is not None: reserve *= 1+.6*float(risk[t])
            reserve = min(reserve, max(0,init-d[:t+1].sum())) if t<4 else reserve
            bu[nc+t] = (reserve,None)
        result = linprog(objective, A_ub=a_ub.tocsr(), b_ub=b_ub,
                         A_eq=A.tocsr(), b_eq=b, bounds=bu, method="highs")
        if not result.success:
            raise RuntimeError(f"Policy optimization failed: {result.message}")
        q = np.maximum(result.x[:nc].reshape(T,C),0)
    return q*scale


def plan_policy(data, origin, start, end, risk_forecast=None,
                min_share=.23, max_share=.47, safety_weeks=1.5,
                scenario_count=12, initial_stock="window_mean", max_order_weeks=1.6):
    """Return implementable fixed weekly orders from one information cutoff.

    Future demand is a published schedule and can be used. No future supply,
    prices, news, shipping, tariffs or outcomes are used here. `risk_forecast`
    may contain material/week/disruption_prob/price_forecast. Price forecasts
    scale supplier landed costs relative to the last observed benchmark.
    """
    if origin >= start:
        raise ValueError("Forecast origin must precede ordering window")
    history = data["supply_history"].query("week <= @origin")
    rows = []
    rf = _policy_frame(risk_forecast) if risk_forecast is not None else None
    for mat, suppliers in data["suppliers"].groupby("material", sort=True):
        suppliers = suppliers.reset_index(drop=True)
        d = data["demand"].loc[data["demand"].material.eq(mat) & data["demand"].week.between(start,end)].sort_values("week").demand_units.to_numpy(float)
        if len(d) != end-start+1: raise ValueError(f"Missing demand: {mat}")
        h = history[history.material.eq(mat)]
        cap, lead, cost, weights, cur_cost = _scenarios(h, suppliers, origin, start, len(d), scenario_count)
        reference = float(cur_cost.mean())
        risk = None
        if rf is not None:
            f = rf.loc[rf.material.eq(mat) & rf.week.between(start,end)].sort_values("week")
            if len(f)==len(d):
                risk = f.disruption_prob.to_numpy(float)
                if "price_forecast" in f:
                    p = data["outcomes"].loc[data["outcomes"].material.eq(mat) & data["outcomes"].week.le(origin)].sort_values("week").benchmark_price.iloc[-1]
                    ratio = np.clip(f.price_forecast.to_numpy(float)/p,.6,1.7)
                    cost = cost*ratio[None,:,None]
        # Enforce tariff announcements visible at the forecast origin.
        if "tariffs" in data:
            tariffs = data["tariffs"]
            for cidx,country in enumerate(suppliers.country):
                tr = tariffs[tariffs.material.eq(mat) & tariffs.country.eq(country) & tariffs.week.le(origin)].sort_values("week")
                if len(tr):
                    current = tr.iloc[-1].tariff_rate
                    ann = tr[tr.announced_rate.notna() & tr.announced_effective_week.gt(origin)]
                    if len(ann):
                        announced = ann.iloc[-1]
                        for t,week in enumerate(range(start,end+1)):
                            if week>=announced.announced_effective_week:
                                cost[:,t,cidx] *= (1+announced.announced_rate)/(1+current)
        initial = _initial_stock(data["demand"],mat,start,end,origin,initial_stock)
        quantity = _optimize_orders(d, initial, cap,lead,cost,weights,reference,
                                    suppliers.normal_capacity_units.to_numpy(float),
                                    min_share,max_share,safety_weeks,risk,max_order_weeks)
        for t,week in enumerate(range(start,end+1)):
            total = float(quantity[t].sum())
            shares = quantity[t]/total if total>1e-8 else np.repeat(1/3,3)
            row = dict(week=week,material=mat,order_units=total,
                       **{f"share_{c}":0.0 for c in COUNTRIES})
            for country,share in zip(suppliers.country,shares): row[f"share_{country}"]=float(share)
            rows.append(row)
    return pd.DataFrame(rows)
