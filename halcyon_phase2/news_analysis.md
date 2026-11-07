# News signals and extraction

The supplied news file contains 4,317 articles through week 208, with 231 distinct
headlines. It contains repeated templates and syndicated articles. The extraction
uses headlines to identify event stage and headline plus body to identify the
affected material, country, and shipping lane. Sources and article bodies are
treated as data, not instructions.

## Signals visible at the forecast origin

| Evidence | Scope | Interpretation as of week 208 |
|---|---|---|
| A04272, week 207, and A04298, week 208: Kestria lithium producers flag maintenance delays | KES lithium | Repeated early production warning. This does not establish that an active shock has started. |
| A04284, week 207: congestion builds at Southern route ports; A04306, week 208: Southern route labor talks drag on | SOUTH lane, used by ORI | Early labor/logistics warning affects ORI supplies of lithium, cobalt, copper, and natural gas. No definite strike date is in these latest articles. |
| A04199, A04223, A04248, weeks 203–205: Valdmar tensions; A04262, week 206: warning shots near Northern Sea route | NORTH lane, used by VAL | These warnings are followed by A04290, week 207, stating that calm returned. The unresolved-warning state is therefore cleared. |

No material is labeled disrupted in outcomes for weeks 199–208. That is an
observation available at the origin, not a guarantee about weeks 209–234.
The latest news supports elevated near-term attention to lithium and the ORI
shipping lane. It gives much weaker information about the private window,
which begins 14 weeks after the supplied market-data cutoff.

## Feature interface

```python
from news_features import get_news_features, relevant_news
features = get_news_features(news, suppliers, origin=208, material="lithium")
```

`get_news_features` returns 87 numerical features. It exposes four stages
(early, late, active, resolved) across seven classes (tariff, export, supply,
strike, conflict, sanctions, ongoing). It supplies rolling four- and thirteen-week
counts, recent-stage ages, decayed counts, and unresolved event state. Counts are
weighted by the fraction of the material's eligible suppliers affected. It also
returns unresolved warning and active supplier fractions.

At origin 208, the warning fraction is 2/3 for lithium, 1/3 for cobalt, copper,
and natural gas, and zero for the other materials. These fractions measure
exposure to reported warnings; they are **not disruption probabilities**.

Broad Kestria sanctions apply to every material sold by KES. Material-specific
warnings affect only that material and exporter. Lane news applies to suppliers
on that lane, including both KES and SUN for MERROW. Aurelia is excluded as the
affected supplier when it appears as the regulator imposing a restriction on a
named external exporter. Resolution records clear the appropriate supplier or
lane state. The extractor distinguishes an actual resolution headline from the
negative phrase “no timeline for when flows will return to normal” in a body.

The extractor assigns reported events to stages, rather than treating headlines
as verified outcome labels. For example, an article reporting an accident can
precede the benchmark's disruption label. False alarms and resolutions are kept
as features for the forecasting model to learn from.

## Leakage safeguards and checks

- Every aggregate and state uses only articles with `week <= origin`.
- Fixed matching rules use no future labels, vocabulary fitting, or future counts.
- Duplicate week/headline/body records are removed before counting, so multiple
  outlets repeating the same report do not create extra independent evidence.
- Features from the full news frame at origin 100 were exactly equal to features
  from a separate copy truncated at week 100.
- Latest NORTH resolution, KES lithium warning, and ORI lane warning were inspected
  against their source article IDs.

Input DataFrames should be treated as immutable during a run because parsing and
feature extraction are cached for repeated training horizons. Re-reading updated
CSV files into new DataFrames refreshes the cache. This module is not a probability
model and does not by itself establish out-of-sample accuracy.
