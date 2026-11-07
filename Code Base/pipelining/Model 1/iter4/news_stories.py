"""
news_stories.py - Model 1 iter4 news layer: headline story chains -> onset probabilities.

The news in this world follows fixed templates that escalate before a shock (weeks 1-208):
  shortage  (one supplier)  "<C> <M> producers flag maintenance delays", "Workers at <C> <M> sites
            threaten walkout", "<C> rations water to <M> producers", "Accident halts major <C> <M>
            site", "<C> <M> output forecast cut sharply"             -> 3 of 3 became shocks
  export    (one supplier)  "Lawmakers in <C> debate limits on raw material exports" (country only),
            "<C> ministry calls <M> a strategic priority", "<C> <M> exporters report slower license
            approvals", "<C> drafts restrictions on <M> shipments abroad", "<C> signals <M> export curbs
            within weeks", "Traders brace for <C> <M> export halt"
            rumor end: "<C> officials rule out new restrictions"     -> 3 of 4 became shocks
  strike    (one lane)      "Dockworkers on the <L> vote on contract offer", "Port labor talks drag on
            for the <L>", "Congestion builds at ports on the <L>", "Port waiting times creep up along
            the <L>", "Union sets strike date for <L> ports", "Strike on the <L> looks likely",
            "Shippers rush cargo ahead of <L> strike"
            rumor end: "Deal reached, <L> disruption averted"        -> 2 of 6 became closures
  clash     (one lane)      "Tensions rise between <C> and neighbors" (lane of C), "Naval patrols
            increase near the <L>", "Insurers raise war risk premiums for the <L>", "Warning shots
            fired near the <L>"
            rumor end: "Calm returns near the <L>"                   -> 1 of 3 became closures
  sanction  (one country)   "Diplomatic talks between Aurelia and <C> break down", "<C> currency slips
            as investors grow cautious", "Aurelian lawmakers criticize <C> government", "Aurelian
            officials review <C> trade practices", "Companies told to prepare for <C> sanctions",
            "Sanctions vote on <C> scheduled"                        -> 1 of 1 became an event
A story opens at its first warning, and closes when its target's shock starts (it happened), when
a rumor-end headline appears, or `timeout` weeks after its last warning (it did not happen).

At an origin T, only stories already closed by T are used to estimate, per story type:
  p_happen    (happened + prior_n * prior_p) / (closed stories + prior_n)
  lead pmf    weeks from first warning to onset, smoothed with a flat prior over 1-8 weeks
For each story still open at T (first warning w0, elapsed e = T - w0):
  P(happens | not yet) = p * P(L > e) / (p * P(L > e) + 1 - p)
  q[k] = P(happens | not yet) * P(L = e + k | L > e)     (first onset k weeks after T)
"Lawmakers in <C> debate limits" names no material: its q is split across C's suppliers.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

COUNTRY = {"Kestria": "KES", "Valdmar": "VAL", "Sunda Coast": "SUN", "Orinoque": "ORI",
           "Halvard": "HAL", "Aurelia": "AUR"}
LANE = {"Merrow Strait": "MERROW", "Northern Sea route": "NORTH", "Southern route": "SOUTH",
        "Atlantic route": "ATLANTIC", "domestic rail network": "RAIL"}
MATERIAL = {"lithium": "lithium", "cobalt": "cobalt", "copper": "copper", "aluminum": "aluminum",
            "steel": "steel", "rare earth": "rare_earths", "rare earths": "rare_earths",
            "semiconductor": "semiconductors", "semiconductors": "semiconductors",
            "polymer resin": "polymer_resin", "specialty glass": "specialty_glass",
            "natural gas": "natural_gas"}
C = "(" + "|".join(sorted(COUNTRY, key=len, reverse=True)) + ")"
L = "(" + "|".join(sorted(LANE, key=len, reverse=True)) + ")"
M = "(" + "|".join(sorted(MATERIAL, key=len, reverse=True)) + ")"
COUNTRY_LANE = {"KES": "MERROW", "SUN": "MERROW", "VAL": "NORTH", "ORI": "SOUTH", "HAL": "ATLANTIC",
                "AUR": "RAIL"}

# (story type, role, regex, target kind) ; role: warn | end (rumor resolved)
PATTERNS = [
    ("shortage", "warn", rf"^{C} {M} producers flag maintenance delays", "sup"),
    ("shortage", "warn", rf"^Workers at {C} {M} sites threaten walkout", "sup"),
    ("shortage", "warn", rf"^{C} rations water to {M} producers", "sup"),
    ("shortage", "warn", rf"^Accident halts major {C} {M} site", "sup"),
    ("shortage", "warn", rf"^{C} {M} output forecast cut sharply", "sup"),
    ("export", "warn", rf"^Lawmakers in {C} debate limits on raw material exports", "cty_sup"),
    ("export", "warn", rf"^{C} ministry calls {M} a strategic priority", "sup"),
    ("export", "warn", rf"^{C} {M} exporters report slower license approvals", "sup"),
    ("export", "warn", rf"^{C} drafts restrictions on {M} shipments abroad", "sup"),
    ("export", "warn", rf"^{C} signals {M} export curbs within weeks", "sup"),
    ("export", "warn", rf"^Traders brace for {C} {M} export halt", "sup"),
    ("export", "end", rf"^{C} officials rule out new restrictions", "cty_sup"),
    ("strike", "warn", rf"^Dockworkers on the {L} vote on contract offer", "lane"),
    ("strike", "warn", rf"^Port labor talks drag on for the {L}", "lane"),
    ("strike", "warn", rf"^Congestion builds at ports on the {L}", "lane"),
    ("strike", "warn", rf"^Port waiting times creep up along the {L}", "lane"),
    ("strike", "warn", rf"^Union sets strike date for {L} ports", "lane"),
    ("strike", "warn", rf"^Strike on the {L} looks likely", "lane"),
    ("strike", "warn", rf"^Shippers rush cargo ahead of {L} strike", "lane"),
    ("strike", "end", rf"^Deal reached, {L} disruption averted", "lane"),
    ("clash", "warn", rf"^Tensions rise between {C} and neighbors", "cty_lane"),
    ("clash", "warn", rf"^Naval patrols increase near the {L}", "lane"),
    ("clash", "warn", rf"^Insurers raise war risk premiums for the {L}", "lane"),
    ("clash", "warn", rf"^Warning shots fired near the {L}", "lane"),
    ("clash", "end", rf"^Calm returns near the {L}", "lane"),
    ("sanction", "warn", rf"^Diplomatic talks between Aurelia and {C} break down", "cty"),
    ("sanction", "warn", rf"^{C} currency slips as investors grow cautious", "cty"),
    ("sanction", "warn", rf"^Aurelian lawmakers criticize {C} government", "cty"),
    ("sanction", "warn", rf"^Aurelian officials review {C} trade practices", "cty"),
    ("sanction", "warn", rf"^Companies told to prepare for {C} sanctions", "cty"),
    ("sanction", "warn", rf"^Sanctions vote on {C} scheduled", "cty"),
]
COMPILED = [(k, r, re.compile(p, re.I), t) for k, r, p, t in PATTERNS]
SOURCE_OF = {"shortage": "sup", "export": "sup", "strike": "lane", "clash": "lane", "sanction": "cty"}


def parse_news(news, suppliers):
    """Tagged headlines: week, story, role, source ('sup'|'lane'|'cty'), key (supplier row / lane / country)."""
    sup_ix = {(m, c): i for i, (m, c) in enumerate(zip(suppliers.material, suppliers.country))}
    rows = []
    for wk, h in zip(news.week.values, news.headline.values):
        for kind, role, rx, tgt in COMPILED:
            mt = rx.search(str(h))
            if not mt:
                continue
            g = mt.groups()
            if tgt == "sup":
                key = sup_ix.get((MATERIAL[g[1].lower()], COUNTRY[g[0]]))
                keys = [] if key is None else [key]
                src = "sup"
            elif tgt == "cty_sup":          # country named, material not: all of that country's suppliers
                keys = [i for (m, c), i in sup_ix.items() if c == COUNTRY[g[0]]]
                src = "sup"
            elif tgt == "lane":
                keys, src = [LANE[g[0]]], "lane"
            elif tgt == "cty_lane":
                keys, src = [COUNTRY_LANE[COUNTRY[g[0]]]], "lane"
            else:
                keys, src = [COUNTRY[g[0]]], "cty"
            for k in keys:
                rows.append(dict(week=int(wk), story=kind, role=role, source=src, key=k,
                                 split=len(keys) if tgt == "cty_sup" else 1, headline=h))
            break
    return pd.DataFrame(rows, columns=["week", "story", "role", "source", "key", "split", "headline"])


def build_stories(tags, active_of, t_last_week, timeout=10):
    """Group tagged headlines into stories per (story, source, key). active_of(source, key) -> bool
    array over weeks (index = week - 1) of that target's shock state. Returns DataFrame with
    first, last_warn, closed_week (NaN if open at t_last_week), outcome (1 happened / 0 not / NaN open),
    onset_week."""
    out = []
    for (kind, src, key), g in tags.groupby(["story", "source", "key"], sort=False):
        act = active_of(src, key)
        g = g.sort_values("week")
        cur = None

        def close(story, week, outcome, onset=np.nan):
            story.update(closed_week=week, outcome=outcome, onset_week=onset)
            out.append(story)

        for wk, role, split in zip(g.week, g.role, g.split):
            if cur is not None:
                # did the target's shock start since the last check? (onset after first warning)
                hit = _first_onset(act, cur["first"], min(wk, cur["last_warn"] + timeout))
                if hit is not None and hit <= wk:
                    close(cur, hit, 1, hit); cur = None
                elif wk > cur["last_warn"] + timeout:
                    close(cur, cur["last_warn"] + timeout, 0); cur = None
            if role == "end":
                if cur is not None:
                    close(cur, wk, 0); cur = None
                continue
            if cur is None:
                if wk - 1 < len(act) and act[wk - 1]:
                    continue                      # target already in a shock: not a warning
                cur = dict(story=kind, source=src, key=key, first=wk, last_warn=wk, split=split)
            else:
                cur["last_warn"] = wk
        if cur is not None:
            hit = _first_onset(act, cur["first"], min(t_last_week, cur["last_warn"] + timeout))
            if hit is not None:
                close(cur, hit, 1, hit)
            elif t_last_week >= cur["last_warn"] + timeout:
                close(cur, cur["last_warn"] + timeout, 0)
            else:
                cur.update(closed_week=np.nan, outcome=np.nan, onset_week=np.nan)
                out.append(cur)
    cols = ["story", "source", "key", "first", "last_warn", "split", "closed_week", "outcome", "onset_week"]
    return pd.DataFrame(out, columns=cols)


def _first_onset(act, first_week, until_week):
    """First week in (first_week, until_week] where act turns on (was off the week before)."""
    for w in range(first_week, int(until_week) + 1):
        i = w - 1
        if 0 <= i < len(act) and act[i] and (i == 0 or not act[i - 1]):
            return w
    return None


def story_q(stories, origin, H, prior_p=0.5, prior_n=2.0, max_lead=12, lead_prior_max=12, types=None):
    """Per open story at `origin`: q [H] = P(first onset exactly k weeks after origin).
    Uses only stories closed by `origin`. Returns list of dicts (story, source, key, q, p_now, elapsed)."""
    closed = stories[stories.closed_week.notna() & (stories.closed_week <= origin)]
    opened = stories[(stories["first"] <= origin) & (stories.closed_week.isna() | (stories.closed_week > origin))]
    # a country-wide warning ("Lawmakers in <C> debate limits") opens one story per supplier of C;
    # for the rates it counts once: happened if any of those suppliers was hit
    closed = closed.assign(grp=[f"{st}|{f}|{sp}" if sp > 1 else f"{st}|{src}|{k}|{f}"
                                for st, f, sp, src, k in zip(closed.story, closed["first"], closed.split,
                                                             closed.source, closed.key)])
    closed = closed.sort_values("outcome", ascending=False).drop_duplicates("grp")
    out = []
    if types is not None:
        opened = opened[opened.story.isin(types)]
    for _, s in opened.iterrows():
        c = closed[closed.story == s.story]
        n_hit = float(c.outcome.sum())
        p = (n_hit + prior_n * prior_p) / (len(c) + prior_n)
        leads = (c.onset_week - c["first"]).dropna().values.astype(int)
        pmf = np.zeros(max_lead + 1)                 # flat prior over leads 1..lead_prior_max (weight 1)
        pmf[1:lead_prior_max + 1] = 1.0
        pmf = pmf / pmf.sum()
        for ld in leads:
            if 1 <= ld <= max_lead:
                pmf[ld] += 1.0
        pmf = pmf / pmf.sum()
        e = int(origin - s["first"])
        tail = pmf[e + 1:].sum() if e + 1 <= max_lead else 0.0
        if tail <= 1e-9:                             # overdue: spread over the next 3 weeks
            cond = np.zeros(H); cond[:min(3, H)] = 1.0 / min(3, H); tail = 0.05
        else:
            cond = np.zeros(H)
            for k in range(1, H + 1):
                if e + k <= max_lead:
                    cond[k - 1] = pmf[e + k] / tail
        p_now = p * tail / (p * tail + 1 - p)
        out.append(dict(story=s.story, source=s.source, key=s.key, first=int(s["first"]), elapsed=e,
                        p_type=p, p_now=p_now, q=p_now * cond / s.split, n_closed=len(c)))
    return out
