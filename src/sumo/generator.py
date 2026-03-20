"""
SUMO EV Charging Simulation Batch Runner

Generates diverse charging scenarios by varying:
- Battery capacity distribution (mean, std)
- Charging threshold (needToChargeLevel)
- Saturated charge level
- Random seed

Each run produces: tripinfos.xml, charging_output.xml, stops_output.xml
"""

import os
import sys
import shutil
import subprocess
import itertools
import json
import time
from pathlib import Path
from xml.etree import ElementTree as ET

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUMO_DIR = PROJECT_ROOT / "data" / "sumo"
TUTORIAL_DIR = PROJECT_ROOT / "assets" / "sumo_tutorial" / "5_electric"
OUTPUT_BASE = SUMO_DIR / "sim_outputs"

# SUMO binary — adjust if different on your system
SUMO_HOME = os.environ.get(
    "SUMO_HOME", "C:/Program Files (x86)/Eclipse/Sumo"
)
SUMO_BIN = os.path.join(SUMO_HOME, "bin", "sumo.exe")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")

# ── Parameter Grid ─────────────────────────────────────────────────────────
PARAM_GRID = {
    # Battery capacity: normc(mean, std, min, max)
    "battery_mean": [50000, 75000, 100000],
    "battery_std": [10000, 20000],
    # Charging behavior
    "need_to_charge_level": [0.05, 0.10, 0.20],
    "saturated_charge_level": [0.8, 0.9, 1.0],
    # Random seed for route sampling variation
    "seed": [42, 123, 456],
}


def generate_vtypes_xml(output_path: Path, battery_mean: float,
                        battery_std: float, battery_min: float = 10000,
                        battery_max: float = 150000):
    """Generate vtypes.xml with specified battery distribution."""
    root = ET.Element("additional")
    vtype = ET.SubElement(root, "vType", id="DEFAULT_VEHTYPE")
    ET.SubElement(vtype, "param", key="has.battery.device", value="true")
    
    capacity_dist = f"normc({battery_mean},{battery_std},{battery_min},{battery_max})"
    ET.SubElement(vtype, "param", key="device.battery.capacity", value=capacity_dist)
    
    # Initialize charge level to a low distribution so some vehicles need charging
    charge_mean = battery_mean * 0.25  # Start around 25% SOC
    charge_std = battery_mean * 0.15
    charge_dist = f"normc({charge_mean},{charge_std},1000,{battery_max})"
    ET.SubElement(vtype, "param", key="device.battery.chargeLevel", value=charge_dist)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)


def generate_sumocfg(output_path: Path, run_dir: Path,
                     need_to_charge: float, saturated_charge: float,
                     seed: int):
    """Generate a SUMO config file for one simulation run."""
    # Paths relative to tutorial/5_electric (where network/route files are)
    net_file = str(TUTORIAL_DIR / ".." / "3_wizard" / "osm.net.xml.gz")
    route_file = str(TUTORIAL_DIR / ".." / "4_counts" / "sampled.rou.xml")
    vtypes_file = str(run_dir / "vtypes.xml")
    parking_file = str(TUTORIAL_DIR / "parking2.add.xml")
    cstations_file = str(TUTORIAL_DIR / "cstations.add.xml")

    root = ET.Element("sumoConfiguration")

    # Input
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=net_file)
    ET.SubElement(inp, "route-files", value=route_file)
    ET.SubElement(
        inp, "additional-files",
        value=f"{vtypes_file},{parking_file},{cstations_file}"
    )

    # Output
    out = ET.SubElement(root, "output")
    ET.SubElement(out, "tripinfo-output",
                  value=str(run_dir / "tripinfos.xml"))
    ET.SubElement(out, "chargingstations-output",
                  value=str(run_dir / "charging_output.xml"))
    ET.SubElement(out, "stop-output",
                  value=str(run_dir / "stops_output.xml"))

    # Time
    t = ET.SubElement(root, "time")
    ET.SubElement(t, "begin", value="6:0:0")
    ET.SubElement(t, "end", value="8:0:0")

    # Processing
    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport", value="600")

    # Report
    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "error-log", value=str(run_dir / "errors"))
    ET.SubElement(rep, "duration-log.statistics", value="true")
    ET.SubElement(rep, "no-step-log", value="true")

    # Random seed
    rand = ET.SubElement(root, "random_number")
    ET.SubElement(rand, "seed", value=str(seed))

    # Station finder device
    battery = ET.SubElement(root, "battery")
    ET.SubElement(
        battery, "device.stationfinder.probability", value="1"
    )
    ET.SubElement(
        battery, "device.stationfinder.emptyThreshold", value="0.0"
    )
    ET.SubElement(
        battery, "device.stationfinder.needToChargeLevel",
        value=str(need_to_charge)
    )
    ET.SubElement(
        battery, "device.stationfinder.radius", value="600"
    )
    ET.SubElement(
        battery, "device.stationfinder.reserveFactor", value="1.25"
    )
    ET.SubElement(
        battery, "device.stationfinder.saturatedChargeLevel",
        value=str(saturated_charge)
    )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)


def run_single_simulation(run_id: int, params: dict) -> dict:
    """Execute one SUMO simulation with given parameters."""
    run_dir = OUTPUT_BASE / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save parameters
    with open(run_dir / "params.json", "w") as f:
        json.dump({"run_id": run_id, **params}, f, indent=2)

    # Generate config files
    generate_vtypes_xml(
        run_dir / "vtypes.xml",
        battery_mean=params["battery_mean"],
        battery_std=params["battery_std"],
    )
    cfg_path = run_dir / "run.sumocfg"
    generate_sumocfg(
        cfg_path, run_dir,
        need_to_charge=params["need_to_charge_level"],
        saturated_charge=params["saturated_charge_level"],
        seed=params["seed"],
    )

    # Run SUMO
    t0 = time.time()
    result = subprocess.run(
        [SUMO_BIN, "-c", str(cfg_path)],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0

    success = result.returncode == 0
    info = {
        "run_id": run_id,
        "success": success,
        "elapsed_sec": round(elapsed, 2),
        "params": params,
    }

    if not success:
        info["stderr"] = result.stderr[-500:] if result.stderr else ""
        print(f"  [FAIL] Run {run_id}: {result.stderr[-200:]}")
    else:
        print(f"  [OK]   Run {run_id} completed in {elapsed:.1f}s")

    # Save run info
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return info


def generate_param_combinations(grid: dict) -> list[dict]:
    """Generate all combinations from parameter grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_batch(max_runs: int | None = None):
    """Run batch of SUMO simulations with parameter variations."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    combos = generate_param_combinations(PARAM_GRID)
    if max_runs is not None:
        combos = combos[:max_runs]

    total = len(combos)
    print(f"Starting {total} SUMO simulation runs...")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"SUMO binary: {SUMO_BIN}\n")

    results = []
    for i, params in enumerate(combos):
        print(f"[{i+1}/{total}] battery_mean={params['battery_mean']}, "
              f"need_to_charge={params['need_to_charge_level']}, "
              f"saturated={params['saturated_charge_level']}, "
              f"seed={params['seed']}")
        info = run_single_simulation(i, params)
        results.append(info)

    # Save batch summary
    n_success = sum(1 for r in results if r["success"])
    summary = {
        "total_runs": total,
        "successful": n_success,
        "failed": total - n_success,
        "results": results,
    }
    with open(OUTPUT_BASE / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBatch complete: {n_success}/{total} successful")
    print(f"Results saved to {OUTPUT_BASE / 'batch_summary.json'}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SUMO EV Batch Runner")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Limit number of runs (default: all combinations)")
    parser.add_argument("--test", action="store_true",
                        help="Run single test with default params")
    args = parser.parse_args()

    if args.test:
        print("Running single test simulation...")
        info = run_single_simulation(
            run_id=9999,
            params={
                "battery_mean": 50000, "battery_std": 10000,
                "need_to_charge_level": 0.2,
                "saturated_charge_level": 0.8,
                "seed": 42,
            },
        )
        print(f"\nResult: {'SUCCESS' if info['success'] else 'FAILED'}")
    else:
        run_batch(max_runs=args.max_runs)
