import csv


def _fix_mojibake_once(text: str) -> str:
    for encoding in ("latin-1", "cp1252"):
        try:
            return text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def clean_station_name(name: str) -> str:
    cleaned = name.strip()
    for _ in range(5):
        fixed = _fix_mojibake_once(cleaned)
        if fixed == cleaned:
            break
        cleaned = fixed
    return cleaned


def parse_temperature_c(temp_raw: str, unit: str) -> str:
    value = float(temp_raw.strip().replace(",", "."))
    u = unit.strip().upper()
    if u == "F":
        value = (value - 32.0) * 5.0 / 9.0
    elif u != "C":
        raise ValueError(f"Unsupported temperature unit: {unit!r}")
    return f"{round(value, 1) + 0.0:.1f}"


def main() -> None:
    with open("test-input.csv", "r", encoding="utf-8-sig", newline="") as f:
        data_lines = [
            line for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not data_lines:
        raise ValueError("Input file contains no header or data rows.")

    reader = csv.reader(data_lines, delimiter=";")
    header = [col.strip() for col in next(reader)]
    col_idx = {col: i for i, col in enumerate(header)}

    idx_id = col_idx["reading_id"]
    idx_station = col_idx["station_name"]
    idx_temp = col_idx["temperature_raw"]
    idx_unit = col_idx["unit"]
    idx_date = col_idx["reading_date"]

    rows_by_id: dict[int, tuple[str, str, str]] = {}
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        reading_id = int(row[idx_id].strip())
        station_name = clean_station_name(row[idx_station])
        temperature_c = parse_temperature_c(row[idx_temp], row[idx_unit])
        reading_date = row[idx_date].strip()

        if reading_id not in rows_by_id:
            rows_by_id[reading_id] = (station_name, temperature_c, reading_date)

    with open("test-output.csv", "w", encoding="utf-8", newline="\n") as out:
        out.write("reading_id,station_name,temperature_c,reading_date\n")
        for reading_id in sorted(rows_by_id):
            station_name, temperature_c, reading_date = rows_by_id[reading_id]
            out.write(f"{reading_id},{station_name},{temperature_c},{reading_date}\n")


if __name__ == "__main__":
    main()
