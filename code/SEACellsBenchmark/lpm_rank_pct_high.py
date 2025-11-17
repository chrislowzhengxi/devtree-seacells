#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

SYSTEM = "Lateral_plate_mesoderm"
BASE = Path("/project/imoskowitz/xyang2/chrislowzhengxi/results/ucell/Lateral_plate_mesoderm/qc")

def _ensure_manual_edges(edges: pd.DataFrame, system_tag: str) -> pd.DataFrame:
    edges = edges.copy()
    if system_tag == "Lateral_plate_mesoderm":
        need = ~((edges["x_name"] == "Second heart field") &
                 (edges["y_name"] == "Atrial cardiomyocytes")).any()
        if need:
            new_row = {
                "system": system_tag,
                "x": "L_M22",
                "y": "L_M5",
                "x_name": "Second heart field",
                "y_name": "Atrial cardiomyocytes",
                "edge_type": "Developmental progression",
                "x_number": np.nan,
                "y_number": np.nan,
                "x_id": np.nan,
                "y_id": np.nan,
            }
            edges = pd.concat([edges, pd.DataFrame([new_row])], ignore_index=True)
    return edges


def main():
    # Node table from the graph script: pct_midhigh with low_thr=0.65, high_thr=0.65
    nodes = pd.read_csv(BASE / "per_node_pct_midhigh_0.65_0.65.csv")
    nodes = nodes.rename(columns={"pct_midhigh": "pct_high"})

    # Edge table + manual edge
    edges = pd.read_csv("/project/xyang2/SHH/Qiu_TimeLapse/Holly_desktop/edges_filtered.txt", sep="\t")
    edges = edges.loc[edges["system"] == SYSTEM].copy()
    edges = _ensure_manual_edges(edges, SYSTEM)

    # Merge node percentages onto x and y
    nodes_sub = nodes[["celltype_new", "n_total", "pct_high"]]

    edges = edges.merge(
        nodes_sub.rename(columns={"celltype_new": "x_name",
                                  "n_total": "n_x",
                                  "pct_high": "value_x"}),
        on="x_name",
        how="left",
    )

    edges = edges.merge(
        nodes_sub.rename(columns={"celltype_new": "y_name",
                                  "n_total": "n_y",
                                  "pct_high": "value_y"}),
        on="y_name",
        how="left",
    )

    # Deltas and ranking
    edges["delta"] = edges["value_y"] - edges["value_x"]
    edges["abs_delta"] = edges["delta"].abs()
    edges = edges.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    edges["rank"] = edges.index + 1

    out = BASE / "LPM_pct_high_0.65_edge_ranking.csv"
    edges.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
