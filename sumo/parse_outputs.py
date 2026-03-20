"""
SUMO Simulation Output Parser

Parses tripinfos.xml, charging_output.xml, stops_output.xml
and aggregates into temporal bins per charging station.

Output format per simulation run:
  - station_features: (N_stations, T_bins, F_features) numpy array
  - metadata: simulation parameters + summary statistics
"""

import json
import numpy as np
from pathlib import Path
from lxml import etree

# ── Constants ──────────────────────────────────────────────────────────────
SIM_START = 21600  # 6:00 AM in seconds
SIM_END = 28800    # 8:00 AM in seconds
BIN_SIZE = 300     # 5 minutes in seconds
N_BINS = (SIM_END - SIM_START) // BIN_SIZE  # 24

STATION_IDS = [f"cs{i}" for i in range(53)]
STATION_ID_MAP = {sid: idx for idx, sid in enumerate(STATION_IDS)}
N_STATIONS = len(STATION_IDS)


def time_to_bin(t: float) -> int:
    """Convert simulation time to temporal bin index."""
    bin_idx = int((t - SIM_START) / BIN_SIZE)
    return max(0, min(bin_idx, N_BINS - 1))


def parse_tripinfos(path: Path) -> list[dict]:
    """Parse tripinfos.xml for per-vehicle trip and battery data."""
    trips = []
    for event, elem in etree.iterparse(str(path), events=("end",),
                                        tag="tripinfo"):
        trip = {
            "id": elem.get("id"),
            "depart": float(elem.get("depart")),
            "arrival": float(elem.get("arrival")),
            "duration": float(elem.get("duration")),
            "route_length": float(elem.get("routeLength")),
            "waiting_time": float(elem.get("waitingTime")),
            "time_loss": float(elem.get("timeLoss")),
        }
        # Battery sub-element
        battery_elem = elem.find("battery")
        if battery_elem is not None:
            trip["battery_depleted"] = int(battery_elem.get("depleted", "0"))
            trip["actual_capacity"] = float(
                battery_elem.get("actualBatteryCapacity", "0")
            )
            trip["total_energy_consumed"] = float(
                battery_elem.get("totalEnergyConsumed", "0")
            )
            trip["total_energy_regen"] = float(
                battery_elem.get("totalEnergyRegenerated", "0")
            )
        trips.append(trip)
        elem.clear()
    return trips


def parse_charging_stations(path: Path) -> dict[str, dict]:
    """Parse charging_output.xml for per-station charging events."""
    stations = {}
    for event, elem in etree.iterparse(str(path), events=("end",),
                                        tag="chargingStation"):
        cs_id = elem.get("id")
        total_energy = float(elem.get("totalEnergyCharged", "0"))
        charging_steps = int(elem.get("chargingSteps", "0"))

        vehicles = []
        for veh in elem.findall("vehicle"):
            veh_info = {
                "id": veh.get("id"),
                "energy_charged": float(
                    veh.get("totalEnergyChargedIntoVehicle", "0")
                ),
                "begin": float(veh.get("chargingBegin", "0")),
                "end": float(veh.get("chargingEnd", "0")),
            }
            vehicles.append(veh_info)

        stations[cs_id] = {
            "total_energy": total_energy,
            "charging_steps": charging_steps,
            "vehicles": vehicles,
        }
        elem.clear()
    return stations


def parse_stops(path: Path) -> list[dict]:
    """Parse stops_output.xml for charging stop events."""
    stops = []
    for event, elem in etree.iterparse(str(path), events=("end",),
                                        tag="stopinfo"):
        cs_id = elem.get("chargingStation")
        if cs_id is None:
            elem.clear()
            continue
        stop = {
            "vehicle_id": elem.get("id"),
            "charging_station": cs_id,
            "started": float(elem.get("started")),
            "ended": float(elem.get("ended")),
            "blocked_duration": float(elem.get("blockedDuration", "0")),
        }
        stops.append(stop)
        elem.clear()
    return stops


def aggregate_station_features(trips: list[dict],
                                charging_data: dict[str, dict],
                                stops: list[dict]) -> np.ndarray:
    """
    Aggregate data into (N_stations, N_bins, N_features) array.

    Features per station per time bin:
      0: num_arrivals       - vehicles arriving to charge
      1: avg_charging_dur   - average charging duration (seconds)
      2: total_energy       - total energy charged (Wh)
      3: max_blocked_dur    - maximum blocked/waiting duration
      4: utilization        - fraction of bin time with active charging
      5: avg_soc_arrival    - average SoC at arrival (estimated)
    """
    n_features = 6
    features = np.zeros((N_STATIONS, N_BINS, n_features), dtype=np.float32)

    # Accumulators
    arrival_counts = np.zeros((N_STATIONS, N_BINS), dtype=np.int32)
    charging_durs = [[[] for _ in range(N_BINS)] for _ in range(N_STATIONS)]
    blocked_durs = [[[] for _ in range(N_BINS)] for _ in range(N_STATIONS)]

    # Process stops
    for stop in stops:
        cs_id = stop["charging_station"]
        if cs_id not in STATION_ID_MAP:
            continue
        s_idx = STATION_ID_MAP[cs_id]
        t_bin = time_to_bin(stop["started"])

        arrival_counts[s_idx, t_bin] += 1
        dur = stop["ended"] - stop["started"]
        charging_durs[s_idx][t_bin].append(dur)
        blocked_durs[s_idx][t_bin].append(stop["blocked_duration"])

    # Process charging station data for energy
    for cs_id, cs_data in charging_data.items():
        if cs_id not in STATION_ID_MAP:
            continue
        s_idx = STATION_ID_MAP[cs_id]
        for veh in cs_data["vehicles"]:
            t_bin = time_to_bin(veh["begin"])
            features[s_idx, t_bin, 2] += veh["energy_charged"]

    # Fill feature array
    for s in range(N_STATIONS):
        for t in range(N_BINS):
            # Feature 0: num arrivals
            features[s, t, 0] = arrival_counts[s, t]

            # Feature 1: avg charging duration
            if charging_durs[s][t]:
                features[s, t, 1] = np.mean(charging_durs[s][t])

            # Feature 2: total energy (already filled above)

            # Feature 3: max blocked duration
            if blocked_durs[s][t]:
                features[s, t, 3] = np.max(blocked_durs[s][t])

            # Feature 4: utilization (total charging time / bin size)
            if charging_durs[s][t]:
                total_charging = sum(charging_durs[s][t])
                features[s, t, 4] = min(total_charging / BIN_SIZE, 1.0)

            # Feature 5: avg SoC at arrival (estimated from energy charged)
            # Higher energy charged = lower SoC at arrival
            # Normalize: energy / max_capacity (35000 Wh)
            if charging_durs[s][t]:
                avg_energy = features[s, t, 2] / max(arrival_counts[s, t], 1)
                features[s, t, 5] = 1.0 - min(avg_energy / 35000, 1.0)

    return features


def parse_single_run(run_dir: Path) -> dict | None:
    """Parse all outputs from a single simulation run."""
    tripinfo_path = run_dir / "tripinfos.xml"
    charging_path = run_dir / "charging_output.xml"
    stops_path = run_dir / "stops_output.xml"
    params_path = run_dir / "params.json"

    if not all(p.exists() for p in [tripinfo_path, charging_path, stops_path]):
        return None

    trips = parse_tripinfos(tripinfo_path)
    charging_data = parse_charging_stations(charging_path)
    stops = parse_stops(stops_path)

    features = aggregate_station_features(trips, charging_data, stops)

    # Summary statistics
    n_charging_events = len(stops)
    stations_used = sum(
        1 for cs in charging_data.values() if cs["charging_steps"] > 0
    )
    total_energy = sum(cs["total_energy"] for cs in charging_data.values())
    n_depleted = sum(1 for t in trips if t.get("battery_depleted", 0) > 0)

    # Load params
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)

    return {
        "features": features,  # (53, 24, 6)
        "params": params,
        "stats": {
            "n_trips": len(trips),
            "n_charging_events": n_charging_events,
            "stations_used": stations_used,
            "total_energy_charged": round(total_energy, 2),
            "n_depleted": n_depleted,
            "avg_energy_consumed": round(
                np.mean([t["total_energy_consumed"] for t in trips
                         if "total_energy_consumed" in t]), 2
            ),
        },
    }


def parse_all_runs(output_base: Path | None = None) -> list[dict]:
    """Parse all simulation runs in the output directory."""
    if output_base is None:
        output_base = Path(__file__).resolve().parent / "sim_outputs"

    run_dirs = sorted(output_base.glob("run_*"))
    print(f"Found {len(run_dirs)} simulation runs in {output_base}")

    results = []
    for i, run_dir in enumerate(run_dirs):
        parsed = parse_single_run(run_dir)
        if parsed is not None:
            results.append(parsed)
            if (i + 1) % 10 == 0:
                print(f"  Parsed {i+1}/{len(run_dirs)} runs")

    print(f"Successfully parsed {len(results)}/{len(run_dirs)} runs")
    return results


def save_dataset(results: list[dict], output_path: Path):
    """Save parsed results as numpy arrays for training."""
    features = np.stack([r["features"] for r in results])  # (N, 53, 24, 6)
    params_list = [r["params"] for r in results]
    stats_list = [r["stats"] for r in results]

    np.save(output_path / "station_features.npy", features)
    with open(output_path / "run_metadata.json", "w") as f:
        json.dump(
            {"params": params_list, "stats": stats_list},
            f, indent=2
        )

    print(f"Dataset saved: features shape = {features.shape}")
    print(f"  → {output_path / 'station_features.npy'}")
    print(f"  → {output_path / 'run_metadata.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse SUMO outputs")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Parse single run directory")
    parser.add_argument("--all", action="store_true",
                        help="Parse all runs and save dataset")
    args = parser.parse_args()

    if args.run_dir:
        result = parse_single_run(Path(args.run_dir))
        if result:
            print(f"Features shape: {result['features'].shape}")
            print(f"Stats: {json.dumps(result['stats'], indent=2)}")
        else:
            print("Failed to parse run directory")
    elif args.all:
        results = parse_all_runs()
        if results:
            output_path = Path(__file__).resolve().parent / "sim_outputs"
            save_dataset(results, output_path)
    else:
        # Quick test: parse the test run
        test_dir = Path(__file__).resolve().parent / "sim_outputs" / "run_9999"
        if test_dir.exists():
            result = parse_single_run(test_dir)
            if result:
                f = result["features"]
                print(f"Features shape: {f.shape}")
                print(f"Stats: {json.dumps(result['stats'], indent=2)}")
                print(f"\nNon-zero stations (any activity):")
                for s in range(N_STATIONS):
                    total = f[s, :, 0].sum()
                    if total > 0:
                        print(f"  {STATION_IDS[s]}: "
                              f"{int(total)} arrivals, "
                              f"{f[s, :, 2].sum():.0f} Wh charged")
