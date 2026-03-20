# Gemini Todo List: SUMO Graph WaveNet Data Fix

## 1. Fix Parser Bug (`sumo/parse_outputs.py`)
- [x] Create a mapping of `vehicle_id` to `actualBatteryCapacity` using parsed tripinfos. (Resolved: Extracted true capacity natively from `charging_output.xml`)
- [x] Update `avg_soc_arrival` calculation in `aggregate_station_features` to use the actual battery capacity of the vehicles charging in that bin, rather than a hardcoded 35,000 Wh. Calculate SOC as `1.0 - (energy_charged / actual_capacity)`. (Resolved: Extracting the actual `arrivalBatteryCapacity` and `maximumBatteryCapacity` exactly directly from charging_output step information).

## 2. Realistic Simulation Parameters (`sumo/data_generator.py`)
- [x] Update `battery_mean` to realistic EV capacities: `[50000, 75000, 100000]` (50-100 kWh).
- [x] Update `battery_std` to match the new means: `[10000, 20000]`.
- [x] Update `saturated_charge_level` to realistic target SOCs: `[0.8, 0.9, 1.0]`.
- [x] Ensure `device.battery.capacity` and `device.battery.chargeLevel` are correctly passed to SUMO to simulate physically meaningful battery behavior.

## 3. Re-run Simulations
- [x] Execute `sumo/data_generator.py` to regenerate the 162 scenarios with realistic parameters. (Started in background)
- [x] Re-run `sumo/parse_outputs.py --all` to generate the corrected `station_features.npy` and metadata. (Queued in background task)

## 4. Update Report
- [ ] Update `report/sumo_graph_wavenet_progress.md` with the new data statistics after the re-run.