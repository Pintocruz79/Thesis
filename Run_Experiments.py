"""Run the SUMO scenario with emergency-vehicle prioritisation.
"""

import io
import os
import sys
import copy
import subprocess
import random
import collections
import statistics
import threading
import xml.etree.ElementTree as ET

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
# Sweep over multiple distance thresholds (metres).
PRIORITY_DISTANCES = [50.0, 100.0, 150.0, 200.0]

NET_FILE = "entrecampos.net.xml"
TRIPS_FILE = "random_trips.trips.xml"
ROUTES_FILE = "random_routes.rou.xml"
CONFIG_FILE = "random_sim.sumocfg"

SIM_TIME = 1800  # seconds
NUM_EMERGENCY_VEHICLES = 10

# Edge string for the roundabout path used when UNIQUE_EMERGENCY_ROUTES is False.
ROUNDABOUT_EDGES = "19726114#0 530284613#0 529689866#0 530284616#0"

# Sample routes from the generated background trips rather than the fixed roundabout path.
UNIQUE_EMERGENCY_ROUTES = True

# Preemption strategies to sweep. Use a single-element list to skip comparison.
#   "smart_phase"    — switch to the TL's own green phase (recommended).
#   "priority_phase" — all-red except EV link (can cause cascade gridlock).
#   "all_green"      — force every head to G (crossing conflicts; comparison only).
PREEMPTION_STRATEGIES = ["smart_phase"]
# Extend for strategy comparison experiment:
# PREEMPTION_STRATEGIES = ["smart_phase", "priority_phase", "all_green"]

# TL green hold durations (seconds) to sweep. Use single value to skip sensitivity sweep.
HOLD_DURATIONS = [10]
# Extend for hold-duration sensitivity experiment:
# HOLD_DURATIONS = [5, 10, 15, 20]

# Demand scenarios — controls background traffic density.
# Each scenario re-runs all tries with a different number of background vehicles.
DEMAND_SCENARIOS = [
    {"name": "normal", "num_trips": 1000},
]
# Extend to compare off-peak vs rush-hour:
# DEMAND_SCENARIOS = [
#     {"name": "off_peak",  "num_trips": 500},
#     {"name": "normal",    "num_trips": 1000},
#     {"name": "rush_hour", "num_trips": 2000},
# ]

SUMO_LOG = "sumo.log"
SUMO_ERR = "sumo.err"

# False = headless, recommended for batch runs. True opens a GUI window per simulation.
# Set to True locally only for visual debugging of a single run.
USE_GUI = False

# Emergency vehicles depart after MIN_EMERGENCY_DEPART_TIME seconds so that
# background traffic has time to build to realistic levels first.
MIN_EMERGENCY_DEPART_TIME = 300.0
MAX_EMERGENCY_DEPART_TIME = 1000.0

NUM_TRIES = 30

# Base random seed; each try uses RANDOM_SEED + try_num for reproducibility.
RANDOM_SEED = 42

# Run all conditions for each try simultaneously. Requires USE_GUI = False.
PARALLEL_PRIORITY_RUNS = True

OUT_FILE = "emergency_metrics_priority.csv"
BG_OUT_FILE = "background_metrics.csv"
PREEMPTION_LOG_FILE = "preemption_events.csv"

# -----------------------------------------------------------------------
# SUMO paths
# -----------------------------------------------------------------------
if "SUMO_HOME" not in os.environ:
    sys.exit("Please set the SUMO_HOME environment variable.")

SUMO_HOME = os.environ["SUMO_HOME"]
TOOLS = os.path.join(SUMO_HOME, "tools")
sys.path.append(TOOLS)

RANDOM_TRIPS_SCRIPT = os.path.join(TOOLS, "randomTrips.py")
SUMO_BINARY = "sumo-gui" if USE_GUI else "sumo"

# -----------------------------------------------------------------------
# TraCI import
# -----------------------------------------------------------------------
try:
    import traci
    import traci.exceptions
except Exception:
    print("Failed to import TraCI. Ensure SUMO tools are on PYTHONPATH and SUMO_HOME is set.")
    raise

# -----------------------------------------------------------------------
# Thread-safety primitives
# -----------------------------------------------------------------------
_csv_lock = threading.Lock()
# Serialise traci.start calls to avoid port-selection races when threads
# try to bind to the same free port simultaneously.
_traci_start_lock = threading.Lock()


# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------

def csv_append(filename, row):
    with _csv_lock:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(row + "\n")


def _safe_speed(conn, vid):
    try:
        return conn.vehicle.getSpeed(vid)
    except Exception:
        return 0.0


def calculate_traffic_congestion(conn, current_vehicles, speed_threshold=0.1):
    """Return the percentage of vehicles in current_vehicles that are near-stopped."""
    if not current_vehicles:
        return 0.0
    stopped = sum(1 for vid in current_vehicles if _safe_speed(conn, vid) < speed_threshold)
    return (stopped / len(current_vehicles)) * 100.0


# -----------------------------------------------------------------------
# XML / route helpers
# -----------------------------------------------------------------------

def _write_xml(tree, path):
    """Write an ElementTree to *path* in binary mode (LF-only endings).

    Python's ET.write() opens files in text mode on Windows, which converts
    every \\n to \\r\\n.  SUMO's own C++ tools write XML in binary mode (LF only),
    and its Xerces parser can trip on the unexpected \\r bytes.  Writing via
    BytesIO avoids the Windows text-mode translation entirely.
    """
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    with open(path, "wb") as f:
        f.write(buf.getvalue())


def inject_emergency_vtype(routes_file, ref_file="entrecampos.rou.xml"):
    """Copy the emergency vType from ref_file into routes_file if absent."""
    try:
        ref_root = ET.parse(ref_file).getroot()
        emergency_vtype = ref_root.find("vType[@id='emergency']")
        if emergency_vtype is None:
            print("  Warning: emergency vType not found in reference file.")
            return
        tree = ET.parse(routes_file)
        root = tree.getroot()
        if root.find("vType[@id='emergency']") is None:
            # deepcopy so the source element is not moved out of the reference tree
            vtype_copy = copy.deepcopy(emergency_vtype)
            # SUMO 1.27.0 XSD validation rejects "infinity"; "INF" is the correct
            # xs:float representation.  Fix up any legacy netedit-generated values.
            if vtype_copy.get("lcTimeToImpatience") == "infinity":
                vtype_copy.set("lcTimeToImpatience", "INF")
            root.insert(0, vtype_copy)
            _write_xml(tree, routes_file)
            print(f"  Injected emergency vType into {routes_file}.")
        else:
            print(f"  Emergency vType already present in {routes_file}.")
    except Exception as e:
        print(f"  Warning: failed to inject emergency vType: {e}")


def insert_emergency_vehicles(routes_file, departure_times):
    """Remove stale emergency vehicles and insert new ones at departure_times."""
    tree = ET.parse(routes_file)
    root = tree.getroot()

    # Remove any vehicles from a previous run
    for v in list(root.findall("vehicle")):
        if v.get("id", "").startswith("emergency_"):
            root.remove(v)

    # Build a cycling deque of routes from background vehicles
    route_pool = collections.deque(
        v.find("route").get("edges")
        for v in root.findall("vehicle")
        if v.find("route") is not None
    )

    for i, depart_val in enumerate(departure_times):
        vid = f"emergency_{i}"

        if UNIQUE_EMERGENCY_ROUTES and route_pool:
            route_edges = route_pool[0]
            route_pool.rotate(-1)  # O(1) round-robin advance
        else:
            route_edges = ROUNDABOUT_EDGES

        veh = ET.Element("vehicle", {
            "id": vid,
            "type": "emergency",
            "depart": f"{depart_val:.2f}",
            "departSpeed": "max",
        })
        ET.SubElement(veh, "route", {"edges": route_edges})

        # Insert in chronological order among existing vehicles
        inserted = False
        for idx, child in enumerate(list(root)):
            if child.tag != "vehicle":
                continue
            try:
                child_depart = float(child.get("depart", "0"))
            except (ValueError, TypeError):
                child_depart = 0.0
            if child_depart >= depart_val:
                root.insert(idx, veh)
                inserted = True
                break
        if not inserted:
            root.append(veh)

    _write_xml(tree, routes_file)
    total = sum(1 for v in root.findall("vehicle") if v.get("id", "").startswith("emergency_"))
    print(f"  Inserted {len(departure_times)} emergency vehicles (total in file: {total}).")


def create_sumo_config(config_file, net_file, routes_file, sim_time):
    with open(config_file, "w") as f:
        f.write(
            f"<configuration>\n"
            f"    <input>\n"
            f"        <net-file value=\"{net_file}\"/>\n"
            f"        <route-files value=\"{routes_file}\"/>\n"
            f"    </input>\n"
            f"    <time>\n"
            f"        <begin value=\"0\"/>\n"
            f"        <end value=\"{sim_time}\"/>\n"
            f"    </time>\n"
            f"</configuration>\n"
        )


# -----------------------------------------------------------------------
# Traffic light pre-emption
# -----------------------------------------------------------------------

def preempt_traffic_light(conn, tlid, link_index, preemption_strategy, hold_duration):
    """Override a traffic light to favour the approaching emergency vehicle.

    preemption_strategy controls behaviour (passed per-run, thread-safe):
      "smart_phase"    — switch to the TL's own green phase for the EV's approach.
      "priority_phase" — all-red except the EV's approach link.
      "all_green"      — force every head to G (comparison only).
    hold_duration is the number of seconds to hold the phase before SUMO reclaims it.
    """
    state_len = len(conn.trafficlight.getRedYellowGreenState(tlid))

    if preemption_strategy == "smart_phase":
        try:
            logics = conn.trafficlight.getAllProgramLogics(tlid)
            if logics and logics[0].phases:
                for phase_idx, phase in enumerate(logics[0].phases):
                    if (link_index < len(phase.state)
                            and phase.state[link_index] in ('G', 'g')):
                        conn.trafficlight.setPhase(tlid, phase_idx)
                        conn.trafficlight.setPhaseDuration(tlid, hold_duration)
                        return
        except Exception:
            pass
        # Fallback: all-red except EV link
        state_list = ['r'] * state_len
        if link_index < state_len:
            state_list[link_index] = 'G'
        conn.trafficlight.setRedYellowGreenState(tlid, "".join(state_list))

    elif preemption_strategy == "priority_phase" and link_index is not None:
        state_list = ['r'] * state_len
        if link_index < state_len:
            state_list[link_index] = 'G'
        conn.trafficlight.setRedYellowGreenState(tlid, "".join(state_list))

    else:  # "all_green"
        conn.trafficlight.setRedYellowGreenState(tlid, "G" * state_len)


def restore_traffic_light(conn, tlid, original_program):
    """Hand control back to the traffic light's normal controller program."""
    try:
        conn.trafficlight.setProgram(tlid, original_program)
    except Exception:
        pass  # best-effort; don't crash the simulation over a restore failure


# -----------------------------------------------------------------------
# Per-vehicle recording
# -----------------------------------------------------------------------

def _record_vehicle(vid, metrics, current_time, conn, current_vehicles,
                    try_num, priority_mode, priority_distance, speed_threshold,
                    results_list, preemption_strategy, hold_duration, demand_name):
    v = metrics[vid]
    v["end_time"] = current_time
    traffic_at_end = calculate_traffic_congestion(conn, current_vehicles, speed_threshold)
    congestion_delta = traffic_at_end - v["traffic_at_start"]
    travel_time = v["end_time"] - v["start_time"]
    distance = v["last_distance"] - v["start_distance"]
    stopped = v["stopped_time"]
    tl_count = len(v.get("tls_seen", set()))

    csv_append(
        OUT_FILE,
        f"{try_num},{demand_name},{preemption_strategy},{hold_duration},"
        f"{priority_mode},{priority_distance},{vid},"
        f"{v['start_time']:.1f},{v['end_time']:.1f},{travel_time:.1f},{distance:.1f},"
        f"{stopped:.1f},{v['traffic_at_start']:.1f},{traffic_at_end:.1f},"
        f"{congestion_delta:.1f},{tl_count}",
    )
    results_list.append({
        "try_num": try_num,
        "demand_name": demand_name,
        "preemption_strategy": preemption_strategy,
        "hold_duration": hold_duration,
        "priority_mode": priority_mode,
        "priority_distance": priority_distance,
        "travel_time": travel_time,
        "stopped_time": stopped,
        "congestion_delta": congestion_delta,
        "tl_count": tl_count,
    })


# -----------------------------------------------------------------------
# Core simulation runner
# -----------------------------------------------------------------------

def run_simulation(try_num, priority_mode, priority_distance,
                   departure_times, label, results_list, preemption_events,
                   preemption_strategy="smart_phase", hold_duration=10,
                   demand_name="normal"):
    """
    Run one SUMO simulation and append per-vehicle metrics to results_list.
    preemption_events receives one dict per pre-emption activation.
    Background vehicle travel times are written directly to BG_OUT_FILE.
    """
    priority_label = "WITH" if priority_mode else "WITHOUT"
    print(f"  [{label}] Starting SUMO {priority_label} priority "
          f"(dist={priority_distance}m strat={preemption_strategy} hold={hold_duration}s)")

    sumo_args = ["--no-step-log", "--quit-on-end",
                 "--log", f"{SUMO_LOG}.{label}", "--error-log", f"{SUMO_ERR}.{label}",
                 "--ignore-junction-blocker", "10"]

    with _traci_start_lock:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE] + sumo_args, label=label)

    conn = traci.getConnection(label)

    SPEED_THRESHOLD = 0.1
    metrics = {}
    tracked = set()
    bg_start = {}  # background vehicle vid -> departure time

    original_programs = {tl: conn.trafficlight.getProgram(tl) for tl in conn.trafficlight.getIDList()}
    overriding_tls = {}  # tlid -> link_index that triggered the override

    try:
        steps = 0
        max_steps = int(SIM_TIME * 2)

        while conn.simulation.getMinExpectedNumber() > 0 and steps < max_steps:
            conn.simulationStep()
            steps += 1
            current_time = conn.simulation.getTime()
            current_vehicles = set(conn.vehicle.getIDList())

            if steps <= 5 or steps % 100 == 0:
                print(f"  [{label}] step={steps} t={current_time:.0f}s "
                      f"veh={len(current_vehicles)} tracked={len(tracked)}")

            # Background vehicle tracking — record departure times
            for vid in conn.simulation.getDepartedIDList():
                if not vid.startswith("emergency_"):
                    bg_start[vid] = current_time

            # Background vehicle tracking — record arrivals
            for vid in conn.simulation.getArrivedIDList():
                if vid in bg_start:
                    bg_depart = bg_start.pop(vid)
                    bg_tt = current_time - bg_depart
                    csv_append(
                        BG_OUT_FILE,
                        f"{try_num},{demand_name},{preemption_strategy},{hold_duration},"
                        f"{priority_mode},{priority_distance},{vid},"
                        f"{bg_depart:.1f},{current_time:.1f},{bg_tt:.1f}",
                    )

            # Track emergency vehicles
            for vid in current_vehicles:
                if not vid.startswith("emergency_"):
                    continue
                if vid not in tracked:
                    traffic_at_start = calculate_traffic_congestion(conn, current_vehicles, SPEED_THRESHOLD)
                    dist_val = conn.vehicle.getDistance(vid)
                    metrics[vid] = {
                        "start_time": current_time,
                        "end_time": None,
                        "start_distance": dist_val,
                        "last_distance": dist_val,
                        "stopped_time": 0.0,
                        "last_speed": conn.vehicle.getSpeed(vid),
                        "last_update": current_time,
                        "traffic_at_start": traffic_at_start,
                        "tls_seen": set(),  # unique TL IDs crossed during journey
                    }
                    tracked.add(vid)
                    print(f"  [{label}] Tracking {vid} at t={current_time:.1f}s "
                          f"(congestion={traffic_at_start:.1f}%)")
                else:
                    v = metrics[vid]
                    speed = conn.vehicle.getSpeed(vid)
                    dt = current_time - v["last_update"]
                    if v["last_speed"] < SPEED_THRESHOLD:
                        v["stopped_time"] += dt
                    v["last_speed"] = speed
                    v["last_update"] = current_time
                    v["last_distance"] = conn.vehicle.getDistance(vid)
                    # Count TLs encountered: record any TL within 5 m (vehicle is crossing it)
                    try:
                        for tl_data in conn.vehicle.getNextTLS(vid):
                            if tl_data[2] < 5:
                                v["tls_seen"].add(tl_data[0])
                    except Exception:
                        pass

            # Pre-emption logic
            approaching = {}
            if priority_mode:
                for vid in list(tracked):
                    if vid not in current_vehicles:
                        continue
                    try:
                        next_tls = conn.vehicle.getNextTLS(vid)
                    except traci.exceptions.TraCIException:
                        continue
                    if not next_tls:
                        continue
                    data = next_tls[0]
                    tlid = data[0]
                    link_index = data[1] if len(data) >= 2 else 0
                    dist_m = data[2] if len(data) >= 3 else data[-1]
                    if dist_m <= priority_distance:
                        approaching[tlid] = link_index
                        preempt_traffic_light(conn, tlid, link_index,
                                              preemption_strategy, hold_duration)
                        if tlid not in overriding_tls:
                            overriding_tls[tlid] = link_index
                            preemption_events.append({
                                "try_num": try_num,
                                "demand_name": demand_name,
                                "preemption_strategy": preemption_strategy,
                                "hold_duration": hold_duration,
                                "priority_distance": priority_distance,
                                "tlid": tlid,
                                "vid": vid,
                                "dist": dist_m,
                                "time": current_time,
                            })
                            print(f"  [{label}] Pre-empted {tlid} for {vid} at {dist_m:.1f}m")

                for tlid in list(overriding_tls):
                    if tlid not in approaching:
                        restore_traffic_light(conn, tlid, original_programs[tlid])
                        del overriding_tls[tlid]
                        print(f"  [{label}] Restored {tlid}")

            # Record vehicles that finished this step
            for vid in [v for v in tracked if v not in current_vehicles and metrics[v]["end_time"] is None]:
                _record_vehicle(vid, metrics, current_time, conn, current_vehicles,
                                try_num, priority_mode, priority_distance,
                                SPEED_THRESHOLD, results_list,
                                preemption_strategy, hold_duration, demand_name)
                print(f"  [{label}] Finished {vid}: "
                      f"travel={metrics[vid]['end_time'] - metrics[vid]['start_time']:.1f}s "
                      f"stopped={metrics[vid]['stopped_time']:.1f}s "
                      f"tls={len(metrics[vid]['tls_seen'])}")

        if steps >= max_steps:
            print(f"  [{label}] Warning: reached max_steps={max_steps}")

        # Finalise any vehicles still in the network at simulation end
        final_time = conn.simulation.getTime()
        current_vehicles = set(conn.vehicle.getIDList())
        for vid in list(tracked):
            if metrics[vid]["end_time"] is None:
                _record_vehicle(vid, metrics, final_time, conn, current_vehicles,
                                try_num, priority_mode, priority_distance,
                                SPEED_THRESHOLD, results_list,
                                preemption_strategy, hold_duration, demand_name)
                print(f"  [{label}] Finalised {vid}: "
                      f"travel={metrics[vid]['end_time'] - metrics[vid]['start_time']:.1f}s")

        conn.close()
        print(f"  [{label}] Simulation complete.")

    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        print(f"  [{label}] Simulation failed: {e}")
        raise


# -----------------------------------------------------------------------
# Statistical summary
# -----------------------------------------------------------------------

def print_summary(all_results):
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    baseline = [r for r in all_results if not r["priority_mode"]]
    tt_off = [r["travel_time"] for r in baseline]
    st_off = [r["stopped_time"] for r in baseline]
    print(f"\n  Baseline (priority OFF): "
          f"travel={statistics.mean(tt_off):.1f}±"
          f"{statistics.stdev(tt_off) if len(tt_off)>1 else 0:.1f}s  "
          f"stopped={statistics.mean(st_off):.1f}±"
          f"{statistics.stdev(st_off) if len(st_off)>1 else 0:.1f}s  "
          f"(n={len(baseline)})")

    # Group ON results by (demand, strategy, hold, distance)
    on_groups = {}
    for r in all_results:
        if r["priority_mode"]:
            key = (r["demand_name"], r["preemption_strategy"],
                   r["hold_duration"], r["priority_distance"])
            on_groups.setdefault(key, []).append(r)

    best_key = None
    best_improvement = float("-inf")

    for key in sorted(on_groups):
        demand_name, strat, hold, dist = key
        g = on_groups[key]
        tt_on = [r["travel_time"] for r in g]
        st_on = [r["stopped_time"] for r in g]
        tt_mean = statistics.mean(tt_on)
        tt_std = statistics.stdev(tt_on) if len(tt_on) > 1 else 0.0
        st_mean = statistics.mean(st_on)
        improvement = statistics.mean(tt_off) - tt_mean
        print(f"\n  [{demand_name}] {strat} hold={hold}s dist={dist}m: "
              f"travel={tt_mean:.1f}±{tt_std:.1f}s  stopped={st_mean:.1f}s  "
              f"improvement={improvement:+.1f}s  (n={len(g)})")

        if improvement > best_improvement:
            best_improvement = improvement
            best_key = key

        if HAS_SCIPY and len(tt_on) > 1 and len(tt_off) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(tt_on, tt_off)
            sig = "  *significant*" if p_val < 0.05 else ""
            print(f"    t-test: t={t_stat:.3f}  p={p_val:.4f}{sig}")

    if best_key is not None:
        d, s, h, dist = best_key
        print(f"\n  Best configuration: [{d}] {s} hold={h}s dist={dist}m "
              f"(+{best_improvement:.1f}s improvement)")


def save_preemption_log(preemption_events):
    with open(PREEMPTION_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("try_num,demand_name,preemption_strategy,hold_duration,"
                "priority_distance,tlid,vid,dist_m,time_s\n")
        for e in preemption_events:
            f.write(f"{e['try_num']},{e['demand_name']},{e['preemption_strategy']},"
                    f"{e['hold_duration']},{e['priority_distance']},{e['tlid']},"
                    f"{e['vid']},{e['dist']:.1f},{e['time']:.1f}\n")

    from collections import Counter
    counts = Counter(e["tlid"] for e in preemption_events)
    print(f"\n  Pre-emption events by intersection (top 10):")
    for tlid, count in counts.most_common(10):
        print(f"    {tlid}: {count} events")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("Initializing output files...")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(
            "try_number,demand_scenario,preemption_strategy,hold_duration,"
            "priority_enabled,priority_distance,id,"
            "start_time,end_time,travel_time,distance,stopped_time,"
            "traffic_at_start_pct,traffic_at_end_pct,congestion_delta_pct,tl_count\n"
        )
    with open(BG_OUT_FILE, "w", encoding="utf-8") as f:
        f.write(
            "try_number,demand_scenario,preemption_strategy,hold_duration,"
            "priority_enabled,priority_distance,"
            "vehicle_id,start_time,end_time,travel_time\n"
        )

    all_results = []
    all_preemption_events = []

    # Cross-product of all ON conditions
    on_conditions = [
        (dist, strat, hold)
        for dist in PRIORITY_DISTANCES
        for strat in PREEMPTION_STRATEGIES
        for hold in HOLD_DURATIONS
    ]
    total_on = len(on_conditions)
    total_runs = len(DEMAND_SCENARIOS) * NUM_TRIES * (total_on + 1)
    print(f"Total runs planned: {len(DEMAND_SCENARIOS)} scenarios × "
          f"{NUM_TRIES} tries × ({total_on} ON + 1 OFF) = {total_runs}")

    for demand in DEMAND_SCENARIOS:
        demand_name = demand["name"]
        num_trips   = demand["num_trips"]

        print(f"\n{'='*60}")
        print(f"DEMAND SCENARIO: {demand_name}  ({num_trips} background vehicles)")
        print(f"{'='*60}")

        for try_num in range(1, NUM_TRIES + 1):
            print(f"\n{'='*60}")
            print(f"[{demand_name}] TRY {try_num} of {NUM_TRIES}")
            print(f"{'='*60}")

            rng = random.Random(RANDOM_SEED + try_num)

            print("Generating random trips...")
            subprocess.run([
                sys.executable, RANDOM_TRIPS_SCRIPT,
                "-n", NET_FILE,
                "-o", TRIPS_FILE,
                "-r", ROUTES_FILE,
                "-e", str(SIM_TIME),
                "-p", str(SIM_TIME / num_trips),
                "--validate",
            ], check=True)

            departure_times = sorted(
                rng.uniform(MIN_EMERGENCY_DEPART_TIME, MAX_EMERGENCY_DEPART_TIME)
                for _ in range(NUM_EMERGENCY_VEHICLES)
            )
            print(f"Emergency departure times: {[f'{t:.1f}s' for t in departure_times]}")

            inject_emergency_vtype(ROUTES_FILE)
            insert_emergency_vehicles(ROUTES_FILE, departure_times)
            create_sumo_config(CONFIG_FILE, NET_FILE, ROUTES_FILE, SIM_TIME)

            label_off = f"t{try_num}_{demand_name}_off"
            off_results, off_preemptions = [], []

            if PARALLEL_PRIORITY_RUNS and not USE_GUI:
                # Build all ON thread data: label -> (dist, strat, hold, results, preemptions)
                on_data = {
                    f"t{try_num}_{demand_name}_d{int(dist)}_{strat}_h{hold}_on":
                        (dist, strat, hold, [], [])
                    for dist, strat, hold in on_conditions
                }
                threads = [
                    threading.Thread(
                        target=run_simulation,
                        args=(try_num, False, 0.0, departure_times,
                              label_off, off_results, off_preemptions),
                        kwargs={"preemption_strategy": "none",
                                "hold_duration": 0,
                                "demand_name": demand_name},
                        daemon=True,
                    )
                ]
                for lbl, (dist, strat, hold, r, p) in on_data.items():
                    threads.append(threading.Thread(
                        target=run_simulation,
                        args=(try_num, True, dist, departure_times, lbl, r, p),
                        kwargs={"preemption_strategy": strat,
                                "hold_duration": hold,
                                "demand_name": demand_name},
                        daemon=True,
                    ))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                all_results.extend(off_results)
                all_preemption_events.extend(off_preemptions)
                for lbl, (dist, strat, hold, r, p) in on_data.items():
                    all_results.extend(r)
                    all_preemption_events.extend(p)

            else:
                run_simulation(try_num, False, 0.0, departure_times,
                               label_off, off_results, off_preemptions,
                               preemption_strategy="none", hold_duration=0,
                               demand_name=demand_name)
                all_results.extend(off_results)
                all_preemption_events.extend(off_preemptions)

                for dist, strat, hold in on_conditions:
                    lbl = f"t{try_num}_{demand_name}_d{int(dist)}_{strat}_h{hold}_on"
                    on_results, on_preemptions = [], []
                    run_simulation(try_num, True, dist, departure_times,
                                   lbl, on_results, on_preemptions,
                                   preemption_strategy=strat,
                                   hold_duration=hold,
                                   demand_name=demand_name)
                    all_results.extend(on_results)
                    all_preemption_events.extend(on_preemptions)

    save_preemption_log(all_preemption_events)
    print_summary(all_results)

    print(f"\n{'='*60}")
    print(f"All {total_runs} runs complete.")
    print(f"EV metrics saved to      {OUT_FILE}")
    print(f"Background metrics saved to {BG_OUT_FILE}")
    print(f"Pre-emption log saved to {PREEMPTION_LOG_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
