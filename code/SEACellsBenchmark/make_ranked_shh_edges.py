#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

def make_ranked_table(system: str, base_dir: str):
    base = Path(base_dir)
    edges_path = base / system / f"{system}_edge_filtered_with_shh.csv"
    out_path   = base / system / f"{system}_edges_ranked_shh_ucell_mean.tsv"

    print(f"[LOAD] {edges_path}")
    edges = pd.read_csv(edges_path)

    # Keep the columns that match your Gli1 table
    df = edges.loc[:, ["x_name", "y_name", "n_x", "n_y",
                       "sh_x", "sh_y", "delta", "abs_delta"]].copy()

    df = df.rename(columns={
        "sh_x": "value_x",
        "sh_y": "value_y",
    })

    # Sort by absolute delta and add rank
    df = df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    df.to_csv(out_path, sep="\t", index=False)
    print(f"[SAVE] {out_path}")

if __name__ == "__main__":
    # change base_dir if needed
    make_ranked_table(
        system="Lateral_plate_mesoderm",
        base_dir="/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell",
    )
