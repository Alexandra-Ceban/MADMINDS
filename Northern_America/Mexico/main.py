import csv, glob, json, os, re, sys, unicodedata
from collections import defaultdict

ABBREV = {
    "av": "avenida", "avda": "avenida", "blvd": "boulevard", "cjon": "callejon",
    "and": "andador", "cor": "corredor", "peat": "peatonal", "priv": "privada",
    "ret": "retorno", "int": "interior", "ext": "exterior", "edif": "edificio",
    "depto": "departamento", "dpto": "departamento", "amp": "ampliacion",
    "prol": "prolongacion", "cont": "continuacion", "diag": "diagonal",
    "cda": "cerrada", "cerr": "cerrada", "circ": "circuito", "calz": "calzada",
    "viad": "viaducto", "psj": "pasaje", "perif": "periferico", "nte": "norte",
}


def addr_key(addr):
    s = "".join(c for c in unicodedata.normalize("NFKD", (addr or "").lower()) if not unicodedata.combining(c))
    s = re.sub(r"\bc\.", "calle ", re.sub(r"\bc\.\s*jon\.?\b", "callejon", s))
    return tuple(sorted(ABBREV.get(t, t) for t in re.findall(r"[a-z0-9]+", s)))


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else next(
        (p for p in ("test-input.csv", "input.csv", "train-input.csv") if os.path.isfile(p)),
        next((p for p in sorted(glob.glob("*.csv")) if "output" not in os.path.basename(p).lower()), ""),
    )
    buckets = defaultdict(list)
    with open(in_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = (row.get("record_id") or "").strip()
            if rid:
                buckets[addr_key(row.get("address"))].append(((row.get("source_table") or "").strip(), rid))

    matches = sorted(
        sorted((id_a, id_b))
        for grp in buckets.values()
        for i, (tbl_a, id_a) in enumerate(grp)
        for tbl_b, id_b in grp[i + 1:]
        if tbl_a != tbl_b
    )
    for out in ("test-output.json", "output.json"):
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"matches": matches}, f)


if __name__ == "__main__":
    main()
