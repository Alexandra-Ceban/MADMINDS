# Clean Finland logs and rank towns by distinct neighbors

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DATA_PATH = Path(__file__).with_name("routes.csv")

ALIASES = {
    "abo": "turku",
    "alavo": "alavus",
    "birkala": "pirkkala",
    "bjorneborg": "pori",
    "borga": "porvoo",
    "brahestad": "raahe",
    "esbo": "espoo",
    "fredrikshamn": "hamina",
    "grankulla": "kauniainen",
    "hango": "hanko",
    "helsingfors": "helsinki",
    "hogfors": "karkkila",
    "hyvinge": "hyvinkaa",
    "idensalmi": "iisalmi",
    "ikalis": "ikaalinen",
    "ilmola": "ilmajoki",
    "inga": "inkoo",
    "jakobstad": "pietarsaari",
    "kajana": "kajaani",
    "karleby": "kokkola",
    "kasko": "kaskinen",
    "kervo": "kerava",
    "keuru": "keuruu",
    "kimitoon": "kemionsaari",
    "korsholm": "mustasaari",
    "kumo": "kokemaki",
    "kristinestad": "kristiinankaupunki",
    "kyrkslatt": "kirkkonummi",
    "kronoby": "kruunupyy",
    "lahtis": "lahti",
    "lappo": "lapua",
    "larsmo": "luoto",
    "lojo": "lohja",
    "lovisa": "loviisa",
    "lundo": "lieto",
    "malax": "maalahti",
    "nadendal": "naantali",
    "narpes": "narpio",
    "nykarleby": "uusikaarlepyy",
    "nystad": "uusikaupunki",
    "pargas": "parainen",
    "raumo": "rauma",
    "raseborg": "raasepori",
    "reso": "raisio",
    "sibbo": "sipoo",
    "sjundea": "siuntio",
    "stkarins": "kaarina",
    "stmichel": "mikkeli",
    "tammerfors": "tampere",
    "tavastehus": "hameenlinna",
    "tornea": "tornio",
    "traskanda": "jarvenpaa",
    "tusby": "tuusula",
    "uleaborg": "oulu",
    "ulvsby": "ulvila",
    "vanda": "vantaa",
    "vasa": "vaasa",
    "vichtis": "vihti",
    "villmanstrand": "lappeenranta",
    "virmo": "mynamaki",
    "vora": "voyri",
}

DISPLAY_NAMES = {
    "aanekoski": "Äänekoski",
    "hameenlinna": "Hämeenlinna",
    "hyvinkaa": "Hyvinkää",
    "jarvenpaa": "Järvenpää",
    "jyvaskyla": "Jyväskylä",
    "kemijarvi": "Kemijärvi",
    "kemionsaari": "Kemiönsaari",
    "kankaanpaa": "Kankaanpää",
    "kokemaki": "Kokemäki",
    "kittila": "Kittilä",
    "lempaala": "Lempäälä",
    "leppavirta": "Leppävirta",
    "mantsala": "Mäntsälä",
    "mynamaki": "Mynämäki",
    "narpio": "Närpiö",
    "nurmijarvi": "Nurmijärvi",
    "pieksamaki": "Pieksämäki",
    "riihimaki": "Riihimäki",
    "saarijarvi": "Saarijärvi",
    "seinajoki": "Seinäjoki",
    "siilinjarvi": "Siilinjärvi",
    "sodankyla": "Sodankylä",
    "voyri": "Vöyri",
    "ylojarvi": "Ylöjärvi",
}


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.strip().strip("." ).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", text)


def canonical_key(name: str) -> str:
    key = normalize(name)
    while key in ALIASES:
        key = ALIASES[key]
    return key


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(key, key.capitalize())


def main() -> None:
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    # Exact duplicate
    # route rows are removed before the graph is built; the edge set also
    # collapses repeated and reverse-direction records for the same connection.
    unique_records = {
        (row["log_id"], row["logged_on"], row["office"], row["from_town"],
         row["to_town"], row["route_type"]): row
        for row in rows
    }

    edges = set()
    self_loops = []
    raw_names = set()
    for row in unique_records.values():
        raw_from, raw_to = row["from_town"], row["to_town"]
        raw_names.update((raw_from, raw_to))
        origin, destination = canonical_key(raw_from), canonical_key(raw_to)
        if origin == destination:
            self_loops.append((raw_from, raw_to, row["log_id"]))
            continue
        edges.add(tuple(sorted((origin, destination))))

    neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)

    ranking = sorted(
        ((len(adjacent), town) for town, adjacent in neighbors.items()),
        key=lambda item: (-item[0], display_name(item[1]).casefold()),
    )
    leaders = ranking[:7]
    flag = "SIGMOID_" + "".join(display_name(town)[0] for _, town in leaders).upper()

    print(f"Raw rows: {len(rows)}")
    print(f"Exact duplicate records removed: {len(rows) - len(unique_records)}")
    print(f"Raw town spellings: {len(raw_names)}")
    print(f"Canonical towns: {len(neighbors)}")
    print(f"Distinct undirected connections: {len(edges)}")
    print(f"Self-loops discarded: {len(self_loops)}")
    print("Top 7 by distinct connections:")
    for rank, (degree, town) in enumerate(leaders, 1):
        print(f"{rank}. {display_name(town)} — {degree}")
    print(flag)


if __name__ == "__main__":
    main()
