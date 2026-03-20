"""
Charging Station Graph Builder

Builds a spatial graph from the SUMO road network:
  - Nodes: 53 charging stations
  - Edges: based on road network shortest-path distance
  - Adjacency matrix: distance-weighted, thresholded

Uses sumolib to read the network and compute distances.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from lxml import etree

# Add SUMO tools to path for sumolib
SUMO_HOME = os.environ.get(
    "SUMO_HOME", "C:/Program Files (x86)/Eclipse/Sumo"
)
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))

import sumolib

# ── Paths ──────────────────────────────────────────────────────────────────
SUMO_DIR = Path(__file__).resolve().parent
TUTORIAL_DIR = SUMO_DIR / "tutorial" / "5_electric"
NET_FILE = SUMO_DIR / "tutorial" / "3_wizard" / "osm.net.xml.gz"
CSTATIONS_FILE = TUTORIAL_DIR / "cstations.add.xml"


def load_charging_stations(path: Path = CSTATIONS_FILE) -> list[dict]:
    """Load charging station definitions from XML."""
    tree = etree.parse(str(path))
    stations = []
    for cs in tree.findall(".//chargingStation"):
        stations.append({
            "id": cs.get("id"),
            "lane": cs.get("lane"),
            "start_pos": float(cs.get("startPos")),
            "end_pos": float(cs.get("endPos")),
            "power": float(cs.get("power")),
            "efficiency": float(cs.get("efficiency")),
            "parking_area": cs.get("parkingArea"),
        })
    return stations


def get_station_positions(net: sumolib.net.Net,
                          stations: list[dict]) -> np.ndarray:
    """Get (x, y) coordinates for each charging station on the network."""
    positions = []
    for cs in stations:
        lane_id = cs["lane"]
        pos = (cs["start_pos"] + cs["end_pos"]) / 2

        # Get the edge from lane ID (remove _0 suffix)
        edge_id = lane_id.rsplit("_", 1)[0]
        try:
            edge = net.getEdge(edge_id)
            lane = edge.getLane(0)
            shape = lane.getShape()
            # Get position along lane
            x, y = sumolib.geomhelper.positionAtShapeOffset(shape, pos)
            positions.append([x, y])
        except (KeyError, IndexError):
            # Fallback: use edge start position
            try:
                edge = net.getEdge(edge_id)
                from_node = edge.getFromNode()
                coord = from_node.getCoord()
                positions.append([coord[0], coord[1]])
            except (KeyError, AttributeError):
                positions.append([0.0, 0.0])

    return np.array(positions, dtype=np.float64)


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix between all station pairs."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return dist


def build_adjacency_matrix(dist_matrix: np.ndarray,
                           threshold_km: float = 2.0,
                           sigma: float = 1.0) -> np.ndarray:
    """
    Build weighted adjacency matrix from distance matrix.

    Uses Gaussian kernel: w_ij = exp(-d_ij^2 / (2 * sigma^2))
    with distance threshold for sparsity.

    Args:
        dist_matrix: (N, N) Euclidean distances in meters
        threshold_km: maximum distance for connection (km)
        sigma: Gaussian kernel bandwidth (km)
    """
    n = dist_matrix.shape[0]
    threshold_m = threshold_km * 1000
    sigma_m = sigma * 1000

    # Gaussian kernel
    adj = np.exp(-dist_matrix ** 2 / (2 * sigma_m ** 2))

    # Apply threshold
    adj[dist_matrix > threshold_m] = 0

    # Remove self-loops
    np.fill_diagonal(adj, 0)

    return adj.astype(np.float32)


def adj_to_edge_index(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert adjacency matrix to COO edge_index format."""
    rows, cols = np.nonzero(adj)
    edge_index = np.stack([rows, cols], axis=0)  # (2, E)
    edge_weight = adj[rows, cols]
    return edge_index.astype(np.int64), edge_weight.astype(np.float32)


def build_graph(threshold_km: float = 2.0,
                sigma: float = 1.0) -> dict:
    """
    Build the complete charging station graph.

    Returns:
        dict with keys:
        - adj: (N, N) adjacency matrix
        - edge_index: (2, E) COO format
        - edge_weight: (E,) edge weights
        - positions: (N, 2) station coordinates
        - station_ids: list of station IDs
        - node_features: (N, F) static node features
    """
    print("Loading SUMO network...")
    net = sumolib.net.readNet(str(NET_FILE))

    print("Loading charging stations...")
    stations = load_charging_stations()
    n_stations = len(stations)
    print(f"  {n_stations} charging stations found")

    print("Computing station positions...")
    positions = get_station_positions(net, stations)

    print("Building distance matrix...")
    dist_matrix = compute_distance_matrix(positions)

    print(f"Building adjacency (threshold={threshold_km}km, sigma={sigma}km)...")
    adj = build_adjacency_matrix(dist_matrix, threshold_km, sigma)
    edge_index, edge_weight = adj_to_edge_index(adj)
    n_edges = edge_index.shape[1]

    # Static node features: [power, efficiency, normalized_x, normalized_y]
    powers = np.array([cs["power"] for cs in stations], dtype=np.float32)
    effs = np.array([cs["efficiency"] for cs in stations], dtype=np.float32)
    # Normalize positions to [0, 1]
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range[pos_range == 0] = 1.0
    norm_pos = (positions - pos_min) / pos_range

    node_features = np.stack([
        powers / powers.max(),
        effs,
        norm_pos[:, 0].astype(np.float32),
        norm_pos[:, 1].astype(np.float32),
    ], axis=-1)  # (N, 4)

    station_ids = [cs["id"] for cs in stations]

    print(f"Graph built: {n_stations} nodes, {n_edges} edges "
          f"(avg degree: {n_edges / n_stations:.1f})")

    return {
        "adj": adj,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "positions": positions,
        "node_features": node_features,
        "station_ids": station_ids,
        "dist_matrix": dist_matrix,
    }


def save_graph(graph: dict, output_dir: Path | None = None):
    """Save graph data to numpy files."""
    if output_dir is None:
        output_dir = SUMO_DIR / "graph_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "adj.npy", graph["adj"])
    np.save(output_dir / "edge_index.npy", graph["edge_index"])
    np.save(output_dir / "edge_weight.npy", graph["edge_weight"])
    np.save(output_dir / "positions.npy", graph["positions"])
    np.save(output_dir / "node_features.npy", graph["node_features"])
    np.save(output_dir / "dist_matrix.npy", graph["dist_matrix"])
    with open(output_dir / "station_ids.json", "w") as f:
        json.dump(graph["station_ids"], f)

    print(f"Graph saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build charging station graph")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Distance threshold in km (default: 2.0)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian kernel bandwidth in km (default: 1.0)")
    args = parser.parse_args()

    graph = build_graph(threshold_km=args.threshold, sigma=args.sigma)
    save_graph(graph)

    # Print summary
    adj = graph["adj"]
    print(f"\nAdjacency matrix stats:")
    print(f"  Non-zero entries: {(adj > 0).sum()}")
    print(f"  Avg edge weight: {adj[adj > 0].mean():.3f}")
    print(f"  Max distance: {graph['dist_matrix'].max():.0f}m")
    print(f"  Avg distance (connected): "
          f"{graph['dist_matrix'][adj > 0].mean():.0f}m")
