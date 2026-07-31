#!/usr/bin/env python3
import csv
import json
import math
import sys
from pathlib import Path


def flattened_mueller(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(value)
        for row in data["mueller"]
        for component in row
        for value in component
    ]
    return data, values


root = Path(sys.argv[1])
rows = []
for shape in ("prism", "sphere", "asymmetric"):
    for ka in (25, 30):
        full_data, full = flattened_mueller(
            root / shape / f"ka{ka}" / "full" / "average.json"
        )
        scale = max(abs(value) for value in full)
        for mode in ("64", "auto_v2"):
            test_data, test = flattened_mueller(
                root / shape / f"ka{ka}" / mode / "average.json"
            )
            differences = [a - b for a, b in zip(test, full)]
            full_seconds = float(full_data["timing"]["farfield_s"])
            test_seconds = float(test_data["timing"]["farfield_s"])
            rows.append(
                {
                    "shape": shape,
                    "ka": ka,
                    "spectral_alpha": mode,
                    "farfield_full_s": full_seconds,
                    "farfield_spectral_s": test_seconds,
                    "farfield_speedup": full_seconds / test_seconds,
                    "max_error_over_global_peak": max(
                        abs(value) for value in differences
                    ) / scale,
                    "rms_error_over_global_peak": math.sqrt(
                        sum(value * value for value in differences)
                        / len(differences)
                    ) / scale,
                }
            )

output = root / "summary.csv"
with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(output)
for row in rows:
    print(
        f"{row['shape']:10s} ka={row['ka']:2d} "
        f"alpha={row['spectral_alpha']:>4s} "
        f"speedup={row['farfield_speedup']:.2f}x "
        f"max_error={row['max_error_over_global_peak']:.3e}"
    )
