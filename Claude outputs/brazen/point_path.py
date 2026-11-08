"""
point_path.py - "all-in" files that commit to ONE specific future (weeks 222-234, planned from week 221).

Each path fixes every named event's onset week and length (tariff, cobalt ban, rare-earths ban, Merrow closure,
optional Kestria-wide shutdown) with no background hazard. Probabilities come out at ~0/1 (clipped 0.005/0.995),
cobalt/rare-earth prices follow the path, and Model 3 iter5's LP plans orders on futures that only differ by
capacity noise and jump sizes. Built for P(score > 85), not for the average score.

Usage: python point_path.py PIPELINE_DIR OUT_DIR
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import event_tree as et
import pandas as pd


def path(m_on=None, m_len=5, kes_on=None, cob_on=222, re_on=223, ban_len=6, tar_len=12, bg=0.0):
    sp = et.variant(merrow_p=1.0 if m_on else 0.0, merrow_len=(m_len, m_len), merrow_onset={m_on or 223: 1.0},
                    kes_p=1.0 if kes_on else 0.0)
    sp["merrow"]["lead4_p"] = 0.0
    sp["kes_country"]["onset"] = (kes_on or 230, kes_on or 230)
    sp["tariff_p"], sp["tariff_len"] = 1.0, [tar_len]
    sp["bans"][0].update(p=1.0, onset={cob_on: 1.0}); sp["bans"][1].update(p=1.0, onset={re_on: 1.0})
    sp["ban_len"], sp["ban_jitter"] = [ban_len], False
    sp["background"] = dict(sp["background"], rate=bg)
    return sp


# modal reading of the week-221 tree: tariff from 224 for 12 weeks, cobalt ban from 222 and rare-earths ban from 223
# for 6 weeks each, Merrow closure from 223 (most likely onset) for 5 weeks (middle of the 3-8 range)
PATHS = {
    "path_m223_l5": path(m_on=223, m_len=5),
    "path_m224_l5": path(m_on=224, m_len=5),
    "path_m223_l8": path(m_on=223, m_len=8),
    "path_bigwar": path(m_on=223, m_len=6, kes_on=224),
    "path_calm": path(m_on=None),
}

if __name__ == "__main__":
    info = et.m3.OriginInfo(et.W, et.ORIGIN, et.H)
    main = pd.read_csv(os.path.join(et.PIPE, "..", "..", "submissions", "submission_main_it8.csv"))
    for name, sp in PATHS.items():
        et.build_file(name, sp, main, info, n_big=3000, n_lp=300)
