"""Deterministic, as-of news features for the Halcyon supply-chain data.

No outcome labels or future article statistics are used. Inputs are pandas
DataFrames and should be treated as immutable while this module is in use.
Articles are data, never instructions. Reprints with the same week/headline/body
are deduplicated so syndication does not masquerade as independent evidence.
"""

from __future__ import annotations

import math
import re
import weakref


COUNTRIES = {
    "KES": "kestria", "VAL": "valdmar", "SUN": "sunda coast",
    "ORI": "orinoque", "HAL": "halvard", "AUR": "aurelia",
}
LANES = {
    "MERROW": "merrow strait", "NORTH": "northern sea route",
    "SOUTH": "southern route", "ATLANTIC": "atlantic route",
    "RAIL": "domestic rail network",
}
MATERIALS = {
    "lithium": "lithium", "cobalt": "cobalt", "copper": "copper",
    "aluminum": "aluminum", "steel": "steel", "rare_earths": "rare earth",
    "semiconductors": "semiconductor", "polymer_resin": "polymer resin",
    "specialty_glass": "specialty glass", "natural_gas": "natural gas",
}
TYPES = ("tariff", "export", "supply", "strike", "conflict", "sanctions", "ongoing")
STAGES = ("early", "late", "active", "resolved")
_CACHE = {}
_FEATURE_CACHE = {}


def _classification(headline):
    """Return event class and stage from an article's headline, or None.

    Resolution is determined from headlines, not a naive substring in bodies:
    'no timeline for when flows will return to normal' is NOT a resolution.
    """
    h = headline.lower()
    if any(x in h for x in ("disruption averted", "officials rule out", "supply eases",
                            "flows return to normal", "traffic returns", "calm returns")):
        kind = "export" if "rule out" in h else "ongoing"
        return kind, "resolved"
    if any(x in h for x in ("disruption continues", "disruption drags on", "no quick fix")):
        return "ongoing", "active"
    if any(x in h for x in ("sanctions", "diplomatic talks", "currency slips",
                            "lawmakers criticize")):
        stage = "active" if "take effect" in h else "late" if ("sanctions" in h) else "early"
        return "sanctions", stage
    if any(x in h for x in ("tariff", "duties", "trade dispute", "trade practices")):
        stage = "active" if "take effect" in h else "late" if any(
            x in h for x in ("confirms", "published", "front-load")) else "early"
        return "tariff", stage
    if any(x in h for x in ("export licenses", "halts", "exports", "exporters",
                            "strategic priority", "drafts restrictions", "export curbs",
                            "export halt")) and "accident" not in h:
        stage = "active" if any(x in h for x in ("halts", "abruptly suspends")) else "late" if any(
            x in h for x in ("drafts", "curbs within", "brace for")) else "early"
        return "export", stage
    if any(x in h for x in ("maintenance delays", "threaten walkout", "rations water",
                            "output forecast cut", "production falls", "accident halts")):
        stage = "active" if any(x in h for x in ("production falls", "accident halts")) else "late" if any(
            x in h for x in ("rations water", "forecast cut")) else "early"
        return "supply", stage
    if any(x in h for x in ("dockworkers", "congestion builds", "labor talks", "strike")):
        stage = "active" if "strike shuts" in h else "late" if any(
            x in h for x in ("looks likely", "strike date", "rush cargo")) else "early"
        return "strike", stage
    if any(x in h for x in ("waiting times creep", "tensions rise", "naval patrols",
                            "war risk", "warning shots", "suspended after clashes")):
        stage = "active" if "suspended after" in h else "late" if any(
            x in h for x in ("war risk", "warning shots")) else "early"
        return "conflict", stage
    return None


def _prepare(news):
    cached = _CACHE.get(id(news))
    if cached is not None and cached[0]() is news:
        return cached[1]
    records = []
    clean = news.drop_duplicates(["week", "headline", "body"]).sort_values("week", kind="stable")
    for row in clean.itertuples(index=False):
        classification = _classification(str(row.headline))
        if classification is None:
            continue
        text = (str(row.headline) + " " + str(row.body)).lower()
        countries = tuple(c for c, name in COUNTRIES.items() if name in text)
        # Aurelia is frequently the regulator imposing restrictions on imports.
        # When an external exporter is named, do not flag Aurelia's own supply.
        if len(countries) > 1:
            countries = tuple(c for c in countries if c != "AUR")
        lanes = tuple(l for l, name in LANES.items() if name in text)
        materials = tuple(m for m, name in MATERIALS.items() if name in text)
        records.append({"week": int(row.week), "type": classification[0],
                        "stage": classification[1], "countries": countries,
                        "lanes": lanes, "materials": materials,
                        "headline": str(row.headline), "article_id": str(row.article_id)})
    _CACHE[id(news)] = (weakref.ref(news), records)
    return records


def relevant_news(news, suppliers, origin, material):
    """Return scoped signal records observable at origin, for audit purposes."""
    rows = suppliers[suppliers.material == material]
    result = []
    for record in _prepare(news):
        if record["week"] > origin:
            break
        if record["materials"] and material not in record["materials"]:
            continue
        affected = []
        for supplier in rows.itertuples(index=False):
            if record["lanes"]:
                match = supplier.lane in record["lanes"]
            else:
                match = supplier.country in record["countries"]
            if match:
                affected.append(supplier.country)
        if affected:
            result.append(dict(record, affected_countries=tuple(affected),
                               exposure=len(affected) / len(rows)))
    return result


def get_news_features(news, suppliers, origin, material):
    """Numerical features keyed by news_* using only articles week <= origin.

    Windows include origin. Counts use deduplicated articles and equal supplier
    exposure weights; 1/3 means one of the material's three suppliers is affected.
    State features reset after explicit resolutions. Early/late warnings older
    than 13 weeks and unrefreshed active reports older than 52 weeks expire.
    This encodes what was reported, never whether the report proved true.
    """
    key = (id(news), id(suppliers), int(origin), str(material))
    cached = _FEATURE_CACHE.get(key)
    if cached is not None and cached[0]() is news and cached[1]() is suppliers:
        return dict(cached[2])
    records = relevant_news(news, suppliers, int(origin), material)
    features = {}
    for kind in TYPES:
        for stage in STAGES:
            matching = [r for r in records if r["type"] == kind and r["stage"] == stage]
            for window in (4, 13):
                features[f"news_{kind}_{stage}_{window}w"] = float(sum(
                    r["exposure"] for r in matching if origin - r["week"] < window))
    for stage in STAGES:
        matching = [r for r in records if r["stage"] == stage]
        features[f"news_{stage}_last_age"] = float(min(
            [origin - r["week"] for r in matching] + [52]))
        features[f"news_{stage}_decayed"] = float(sum(
            r["exposure"] * math.exp(-(origin - r["week"]) / 4.0)
            for r in matching if origin - r["week"] <= 26))

    state = {}
    for record in records:
        channel = "lane" if record["lanes"] else "supplier"
        for country in record["affected_countries"]:
            if record["stage"] == "resolved":
                for state_key in list(state):
                    if state_key[0] == country and state_key[1] == channel:
                        del state[state_key]
                continue
            state[(country, channel, record["type"])] = record
    n_suppliers = int((suppliers.material == material).sum())
    for kind in TYPES:
        for stage in ("early", "late", "active"):
            features[f"news_open_{kind}_{stage}"] = float(sum(
                1.0 / n_suppliers for (_, _, event_type), record in state.items()
                if event_type == kind and record["stage"] == stage
                and origin - record["week"] <= (52 if stage == "active" else 13)))
    features["news_warning_supplier_fraction"] = len({country for (country, _, _), r in state.items()
        if r["stage"] in ("early", "late") and origin-r["week"] <= 13}) / n_suppliers
    features["news_active_supplier_fraction"] = len({country for (country, _, _), r in state.items()
        if r["stage"] == "active" and origin-r["week"] <= 52}) / n_suppliers
    _FEATURE_CACHE[key] = (weakref.ref(news), weakref.ref(suppliers), dict(features))
    return features
