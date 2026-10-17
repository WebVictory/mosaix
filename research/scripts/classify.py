import collections
import json
import re
from pathlib import Path

from cats import CATS

LAT = str.maketrans("aceopxyk", "асеорхук")


def norm(s):
    s = s.lower().replace("ё", "е")
    s = re.sub(r"^\s*\d+(\.\d+)*[.)]\s*", "", s)
    s = s.replace("’", "'")
    return s


TR = []
for i, (_name, _parent, tr, _anti, _syn, _) in enumerate(CATS):
    for t in tr:
        TR.append((t, i, re.compile(r"(?<![а-яa-z0-9])" + re.escape(t))))
ANTI = {i: [re.compile(re.escape(a)) for a in c[3]] for i, c in enumerate(CATS)}


def classify(name):
    s = norm(name)
    best = None
    for t, i, rx in TR:
        for m in rx.finditer(s):
            # anti-trigger: if any anti phrase of this cat covers the match position
            bad = False
            for a in ANTI[i]:
                for am in a.finditer(s):
                    if am.start() <= m.start() < am.end():
                        bad = True
            if bad:
                continue
            key = (m.start(), -len(t))
            if best is None or key < best[0]:
                best = (key, i, t)
            break
    if not best:
        return ("прочее", None)
    i = best[1]
    if "(проч" in CATS[i][0]:
        par = CATS[i][1]
        alt = None
        for t, j, rx in TR:
            if j != i and CATS[j][1] == par:
                m = rx.search(s)
                covered = m and any(
                    a.search(s) and any(am.start() <= m.start() < am.end() for am in a.finditer(s)) for a in ANTI[j]
                )
                if m and not covered and (alt is None or (m.start(), -len(t)) < alt[0]):
                    alt = ((m.start(), -len(t)), j, t)
        if alt:
            return (CATS[alt[1]][0], alt[2])
    return (CATS[i][0], best[2])


if __name__ == "__main__":
    r = json.loads(Path("rows.json").read_text())
    out = []
    for x in r:
        c, t = classify(x["name"])
        x["cat"] = c
        x["trig"] = t
        out.append(x)
    Path("classified.json").write_text(json.dumps(out, ensure_ascii=False))
    cnt = collections.Counter(x["cat"] for x in out)
    serv = sum(v for k, v in cnt.items() if k.startswith("служебная"))
    work = len(out) - serv
    other = cnt["прочее"]
    print(
        "total",
        len(out),
        "service",
        serv,
        "work",
        work,
        "other",
        other,
        "coverage %.1f" % (100 * (work - other) / work),
    )
    for k, v in cnt.most_common():
        print(v, k)
