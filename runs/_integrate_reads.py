"""Fold the finish-all notebook's result zips into the repository, refusing what does not prove itself.

The notebook ships each gate only when the gate is complete, but "complete" is a statement about
files existing, not about them agreeing. So this script re-derives, before it writes anything, the
two promises the notebook's own results cell asserts: a memo-free read has to reproduce the
certificate the archived arm logged beside it, and a union of two disjoint suites has to stay the
minimum of its halves. A zip whose numbers disagree stays in the browser; the repository is not a
place for a read of the wrong weights.

A gate that *trains* an arm ships three kinds of file, and the arm's own JSON is the one the other
two are measured against - so it is checked first, against the budget and the suite the plan asked
for, and then handed to the reads as their expectation. Reading that expectation from disk instead
would find nothing (the arm has not landed yet) and silently check no read at all.

    python runs/_integrate_reads.py                # the gates data/ still owes, from Downloads
    python runs/_integrate_reads.py --check        # report only, write nothing
    python runs/_integrate_reads.py --only end14   # one gate, when another zip is from an older run
"""
import argparse, hashlib, json, os, pathlib, sys, tempfile, zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "colab"))
import reads_plan as rp
import build_reads_notebooks as gen

REPO = pathlib.Path(__file__).resolve().parent.parent
DOWNLOADS = pathlib.Path(os.path.expanduser("~")) / "Downloads"
# gate name -> the fragment the browser sees in the filename, from the generator that ships them
FRAGMENT = dict(gen.GATES)
GATES = tuple(FRAGMENT[g] for g, _f in gen.GATES)
KEYS = ("worst_draw_rate", "best_draw_rate", "random_draw_rate_exact", "n_roots_drawn_worst")
REQUIRED = KEYS + ("ckpt", "roots", "deterministic_orders", "order_mean", "order_min",
                   "order_max", "adaptive_games_per_root", "adaptive_mean", "adaptive_min")


def owed_gates():
    """The gates this repository still owes work for, spelled as the browser spells their zips.

    The three read gates travel together: the notebook ships each when its group completes, so a
    plan that owes any read beside a Downloads folder missing one of the three is a session that
    died early, and the script says so instead of ingesting a fraction of a run. An arm gate stands
    alone, because the depth it trains is the only work behind it - which is what lets a notebook
    that owes zero reads still deliver its arms.
    """
    want = set()
    if rp.pending():
        want |= {FRAGMENT[g] for g in rp.GROUPS}
    want |= {FRAGMENT[r["gate"]] for r in rp.arm_rows()}
    return want


def gate_zips(wanted, slug="finishall"):
    """The gate zips in Downloads, one per gate the caller asked for."""
    found = {}
    for gate in sorted(wanted):
        name = "c4_%s_%s_results.zip" % (slug, gate)
        if not (DOWNLOADS / name).is_file():
            raise SystemExit("%s no esta en %s: esa seccion de la cola nunca se descargo"
                             % (name, DOWNLOADS))
        found[gate] = DOWNLOADS / name
    return found


def rows_by_out(rows):
    out = {}
    for r in rows:
        out[r["out"]] = r
    return out


def expectation(row, arms=None):
    """What the archived arm already claims, exactly as the notebook's results cell reads it.

    `arms` holds the arm JSONs a gate shipped, by path: an arm trained inside the same zip is not
    on disk yet, and its own reads still have to be measured against it.
    """
    e = rp.expected(row, (arms or {}).get(row["arm_json"]))
    if not e:
        return None
    return {"cert": e["cert"], "suite": e["suite"], "comparable": e["comparable"]}


def check_arm(row, d):
    """Refusals for an arm JSON a gate ships: is this the arm the plan asked for?

    The arm is the expectation its two reads are measured against, so an arm trained under another
    budget would make both reads agree with it and still not be the published protocol. The budget
    comes off the archives through `reads_plan.arm_budget`, never off a table typed here.
    """
    a = d.get("args", {})
    bad = []
    if a.get("empties") != row["E"] or a.get("seed") != row["seed"]:
        bad.append("el plan pidio %d vacias semilla %d y el archivo dice %s/%s"
                   % (row["E"], row["seed"], a.get("empties"), a.get("seed")))
    if (a.get("roots_file") or "").replace("\\", "/") != row["suite_file"]:
        bad.append("certifico sobre %s, no sobre %s" % (a.get("roots_file"), row["suite_file"]))
    budget = rp.arm_budget(row["E"])
    off = [f for f in rp.ARM_FLAGS if a.get(f) != budget[f]]
    if off:
        bad.append("presupuesto distinto del publicado en %s" % ", ".join(off))
    if "final" not in d.get("repair", {}):
        bad.append("no trae repair.final, que es lo que sus dos lecturas tienen que reproducir")
    return bad


def check_reads(shipped, rows, arms=None):
    """Return (lines, failures): the notebook's two promises re-run over what the zips carry."""
    lines, fail = [], []
    got = {}
    for path, row in rows_by_out(rows).items():
        raw = shipped.get(path)
        disk = REPO / path
        if raw is None and not disk.is_file():
            continue
        try:
            v = json.loads(raw.decode("utf-8")) if raw is not None else json.loads(
                disk.read_text(encoding="utf-8"))
        except Exception as exc:
            fail.append("%s: no es JSON leible (%s)" % (path, exc))
            continue
        missing = [k for k in REQUIRED if k not in v]
        if missing:
            fail.append("%s: le faltan las llaves %s" % (path, ",".join(missing)))
            continue
        got[path] = (row, v)
        tag = "%-16s %-22s on %s" % (row["arm"], row["kind"], row["suite"])
        line = ("W %.3f B %.3f D %.4f | raices a tablas %d/%d | ordenes %d (%.3f-%.3f)"
                % (v["worst_draw_rate"], v["best_draw_rate"], v["random_draw_rate_exact"],
                   v["n_roots_drawn_worst"], v["roots"], v["deterministic_orders"],
                   v["order_min"], v["order_max"]))
        exp = expectation(row, arms)
        if exp and exp["comparable"]:
            c = exp["cert"]
            same = all(abs(c[k] - v[k]) < 1e-9 for k in KEYS)
            line += " | el loop decia %s -> %s" % (
                ",".join("%.3f" % c[k] if isinstance(c[k], float) else str(c[k]) for k in KEYS),
                "COINCIDE" if same else "NO COINCIDE")
            if not same:
                fail.append("%s: estos pesos no son los del brazo %s" % (path, row["arm"]))
        lines.append("  %s %s%s" % (tag, line, "  [del zip]" if raw is not None else
                                    "  [ya en data]"))

    # sec:union's identity, over the three suites whether they came from the zip or were archived
    by_arm = {}
    for path, (row, v) in got.items():
        if row["kind"].startswith("union-"):
            tier = "repaired" if "repaired" in row["kind"] else "base"
            by_arm.setdefault(row["arm"], {}).setdefault(tier, {})[row["suite"]] = v
    for arm, tiers in sorted(by_arm.items()):
        for tier, half in sorted(tiers.items()):
            if not all(t in half for t in ("A", "B", "U")):
                continue
            u, a, b = half["U"], half["A"], half["B"]
            ok = (abs(u["worst_draw_rate"] - min(a["worst_draw_rate"], b["worst_draw_rate"]))
                  < 1e-12
                  and abs(u["random_draw_rate_exact"]
                          - (a["random_draw_rate_exact"] + b["random_draw_rate_exact"]) / 2)
                  < 1e-12
                  and u["n_roots_drawn_worst"]
                  == a["n_roots_drawn_worst"] + b["n_roots_drawn_worst"]
                  and u["roots"] == a["roots"] + b["roots"])
            lines.append("  %-16s %s identidad W_U = min(W_A, W_B): %s  (W %.2f/%.2f/%.2f, "
                         "raices %d+%d=%d)" % (arm, tier, "OK" if ok else "ROTO",
                                               a["worst_draw_rate"], b["worst_draw_rate"],
                                               u["worst_draw_rate"], a["n_roots_drawn_worst"],
                                               b["n_roots_drawn_worst"], u["n_roots_drawn_worst"]))
            if not ok:
                fail.append("%s %s: la union no es el minimo de sus dos mitades" % (arm, tier))
    return lines, fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--only", default=",".join(sorted(owed_gates())),
                    help="comma separated gates to ingest, in case one zip is from an older run")
    args = ap.parse_args()

    owed_g = owed_gates()
    if not owed_g:
        print("data/ no debe ninguna lectura ni ningun brazo: no hay zip que integrar")
        return 0
    wanted = [g.strip() for g in args.only.split(",") if g.strip()]
    stale = [g for g in wanted if g not in owed_g]
    if stale:
        raise SystemExit("%s no es una puerta que data/ deba ahora; las que debe son %s"
                         % (", ".join(stale), ", ".join(sorted(owed_g))))
    zips = gate_zips(wanted)
    read_rows = rp.pending()
    # every row a planned arm owes, not just the repair: its two reads live here rather than in
    # `reads()`, which only knows depths that already have an archived arm
    arm_rows = rp.arm_rows()
    owed = rows_by_out(read_rows + arm_rows)
    shipped, archives, logs, seen = {}, {}, {}, set()
    for gate, path in sorted(zips.items()):
        z = zipfile.ZipFile(path)
        print("%s: %d entradas leidas" % (path.name, len(z.namelist())))
        for name in z.namelist():
            data = z.read(name)
            if name.startswith("data/c4_endgame_roots"):
                # checked inside the loop, per zip: a deck that is wrong in one gate stays wrong
                # for the reads that gate produced, even if another gate carries a good copy of the
                # same file. Merging first would let a good zip launder a bad one.
                disk = REPO / name
                same = disk.is_file() and hashlib.sha256(disk.read_bytes()).hexdigest() == hashlib.sha256(
                    data).hexdigest()
                if name not in seen:
                    print("  suite %-52s %s" % (name, "identica a la confirmada" if same
                                                else "DISTINTA -> no se toca la del repositorio"))
                    seen.add(name)
                if not same:
                    raise SystemExit("%s trae %s, que no es la suite confirmada: la lectura no es "
                                     "del mismo mazo que la comparacion publicada" % (path.name, name))
            elif name.startswith("data/"):
                if name not in owed:
                    raise SystemExit("%s trajo %s, que la cola no debe: o el zip es de otra "
                                     "generacion o el plan cambio" % (path.name, name))
                if owed[name]["kind"] != "arm-repair":
                    shipped[name] = data
                    continue
                try:
                    d = json.loads(data.decode("utf-8"))
                except Exception as exc:
                    raise SystemExit("%s no es JSON leible (%s)" % (name, exc))
                bad = check_arm(owed[name], d)
                if bad:
                    raise SystemExit("%s: %s -> nada se escribio" % (name, "; ".join(bad)))
                archives[name] = (d, data)
                print("  brazo %-52s mazo y presupuesto publicados comprobados" % name)
            else:
                logs[name] = data

    lines, fail = check_reads(shipped,
                              read_rows + [r for r in arm_rows if r["kind"] != "arm-repair"],
                              {name: d for name, (d, _raw) in archives.items()})
    print("\n=== las dos promesas de la celda de resultados, re-ejecutadas aqui ===")
    for l in lines:
        print(l)
    if fail:
        print()
        for f in fail:
            print("  NO VALE: " + f)
        raise SystemExit("una lectura no reproduce lo que el brazo certifico: nada se escribio")
    print("\n  %d lecturas validadas, %d brazos, 0 rechazos" % (len(shipped), len(archives)))

    if args.check:
        print("-- check: nada se escribio")
        return 0
    written = 0
    payloads = dict(shipped)
    payloads.update({name: raw for name, (_d, raw) in archives.items()})
    for name, data in sorted(payloads.items()):
        dest = REPO / name
        if dest.is_file() and dest.read_bytes() == data:
            print("  ya estaba %s" % name)
            continue
        if dest.is_file():
            raise SystemExit("%s ya existe con otros bytes: no se pisa una medicion archivada"
                             % name)
        dest.write_bytes(data)
        print("  escrito   %s (%d bytes)" % (name, len(data)))
        written += 1
    for name, data in sorted(logs.items()):
        dest = REPO / name
        if dest.is_file() and dest.read_bytes() == data:
            continue
        if dest.is_file():
            print("  log distinto ya Archivado, se deja: %s" % name)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written += 1
    still = [r for r in rp.pending() + rp.arm_rows() if not rp.exists(r["out"])]
    print("\n%d archivos nuevos | la cola que queda: %d jobs" % (written, len(still)))
    print("regenera el notebook con python colab/build_reads_notebooks.py para que la cola "
          "refleje lo que ya esta en data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
