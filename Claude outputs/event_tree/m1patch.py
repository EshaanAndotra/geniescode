"""
m1patch.py - one extra option for Model 1 iter6's scenario layer (source_overrides), applied in memory.

A negative p_onset, e.g. ("sup:cobalt/VAL", -1.0, 1, 13, 0), rescales the model's OWN onset curve for that source so
the event starts inside the window with probability |p_onset|, keeping the model's timing. (A positive p_onset spreads
the onset evenly over first_h..last_h, which also moves the timing.) Pipeline files are not modified.
"""
import inspect, textwrap

OLD = """            if ks:
                q_new[np.array(ks) - 1] = p_on / len(ks)
"""
NEW = """            if p_on < 0:
                q_new = qq[i] * (-p_on / max(float(qq[i].sum()), 1e-12))
            elif ks:
                q_new[np.array(ks) - 1] = p_on / len(ks)
"""


def apply(m1):
    cls = m1.DisruptionModel
    if getattr(cls, "_certain_patch", False):
        return
    src = inspect.getsource(cls._source_probs)
    assert src.count(OLD) == 1, "Model 1 override code changed; patch not applied"
    ns = {}
    exec(compile(textwrap.dedent(src.replace(OLD, NEW)), "model1_source_probs_patched", "exec"), m1.__dict__, ns)
    cls._source_probs = ns["_source_probs"]
    cls._certain_patch = True
