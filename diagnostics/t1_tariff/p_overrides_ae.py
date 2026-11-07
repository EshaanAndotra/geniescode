"""
p_overrides.py - rule-based floors on disruption_prob, applied after Model 1.

tariff_floor: an announced or freshly enacted tariff counts as a disrupted week for its material.
Every tariff in weeks 1-221 lasted at least 10 weeks (VAL cobalt 15-26, KES aluminum / lithium /
semiconductors 122-131, VAL rare earths 180-197) and every announced one was enacted. Model 1's
duration curve instead decays the probability within the first 10 weeks (0.97 -> 0.73 for KES
aluminum at weeks 228-234). Flooring at 0.95 from the effective week through effective + 9 lowered
the Brier score in all 13 material-windows planned from origins 12-14, 118-121 and 178-179, and
raised it in none.

    import p_overrides
    sub = p_overrides.tariff_floor(sub, data)        # data = model1.load_data(...)
"""
import numpy as np
import pandas as pd

FLOOR = 0.95
MIN_WEEKS = 10


def tariff_floor(sub, data, windows=((209, 221), (222, 234)), floor=FLOOR, min_weeks=MIN_WEEKS):
    """Raise disruption_prob to `floor` on weeks [eff, eff + min_weeks - 1] for every tariff that is announced
    (effective after the origin) or in force with an onset less than min_weeks before it. Each window uses only
    tariffs.csv rows up to its origin = min(start - 1, last data week)."""
    sub = sub.copy()
    tar = data["tariffs"].merge(data["suppliers"][["material", "country", "base_tariff"]], on=["material", "country"])
    last = int(tar.week.max())
    parts = sub.id.str.extract(r"^w(\d+)_(.+)$")
    wk, mat = parts[0].astype(int), parts[1]
    for start, end in windows:
        origin = min(start - 1, last)
        seen = tar[tar.week <= origin]
        for (m, c), g in seen.groupby(["material", "country"]):
            g = g.sort_values("week")
            now = g.iloc[-1]
            eff = None
            if np.isfinite(now.announced_effective_week) and now.announced_effective_week > origin \
                    and now.announced_rate > now.base_tariff + 1e-9:
                eff = int(now.announced_effective_week)
            elif now.tariff_rate > now.base_tariff + 1e-9:
                calm = g[g.tariff_rate <= g.base_tariff + 1e-9]
                eff = int(calm.week.max()) + 1 if len(calm) else int(g.week.min())
            if eff is None:
                continue
            sel = (mat == m) & wk.between(max(start, eff), min(end, eff + min_weeks - 1))
            sub.loc[sel, "disruption_prob"] = np.maximum(sub.loc[sel, "disruption_prob"], floor)
    return sub
