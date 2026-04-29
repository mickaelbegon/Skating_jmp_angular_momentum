"""Command-line tools for batch skating-flight simulations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skating_aerial_alignment.simulation import (
    FlightSimulationParameters,
    PDControllerConfiguration,
    SkaterFlightSimulator,
)
from skating_aerial_alignment.visualization import launch_app


def _to_serializable(value: Any) -> Any:
    """Recursively convert dataclass-friendly values into JSON-friendly objects."""

    if is_dataclass(value):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _controller_from_payload(payload: dict[str, Any]) -> PDControllerConfiguration:
    """Build a controller configuration from a JSON payload."""

    return PDControllerConfiguration(
        proportional_gains=tuple(float(value) for value in payload["proportional_gains"]),
        derivative_gains=tuple(float(value) for value in payload["derivative_gains"]),
        torque_limits=tuple(float(value) for value in payload["torque_limits"]),
    )


def _parameters_from_payload(payload: dict[str, Any]) -> FlightSimulationParameters:
    """Build simulation parameters from a JSON payload."""

    parameters = FlightSimulationParameters()
    if "controller" in payload:
        parameters = replace(parameters, controller=_controller_from_payload(payload["controller"]))
    for field_name in (
        "angular_velocity_rps",
        "initial_trunk_angles_deg",
        "initial_trunk_velocity_deg_s",
    ):
        if field_name in payload:
            payload[field_name] = tuple(float(value) for value in payload[field_name])
    if "sample_count" in payload:
        payload["sample_count"] = int(payload["sample_count"])
    if "stabilize_trunk" in payload:
        payload["stabilize_trunk"] = bool(payload["stabilize_trunk"])
    return replace(parameters, **payload)


def _parameters_from_arguments(arguments: argparse.Namespace) -> FlightSimulationParameters:
    """Build simulation parameters from parsed command-line arguments."""

    controller = PDControllerConfiguration(
        proportional_gains=tuple(float(value) for value in arguments.kp),
        derivative_gains=tuple(float(value) for value in arguments.kd),
        torque_limits=tuple(float(value) for value in arguments.torque_limits),
    )
    return FlightSimulationParameters(
        angular_velocity_rps=tuple(float(value) for value in arguments.sigma_rps),
        takeoff_vertical_velocity=float(arguments.vertical_velocity),
        backward_horizontal_velocity=float(arguments.backward_velocity),
        somersault_tilt_deg=float(arguments.somersault_tilt_deg),
        inward_tilt_deg=float(arguments.inward_tilt_deg),
        initial_trunk_angles_deg=tuple(
            float(value) for value in arguments.initial_trunk_angles_deg
        ),
        initial_trunk_velocity_deg_s=tuple(
            float(value) for value in arguments.initial_trunk_velocity_deg_s
        ),
        stabilize_trunk=bool(arguments.stabilize_trunk),
        controller=controller,
        sample_count=int(arguments.sample_count),
    )


def _result_summary(
    simulator: SkaterFlightSimulator,
    parameters: FlightSimulationParameters,
    result,
    *,
    label: str | None,
) -> dict[str, Any]:
    """Return a compact summary of one simulated scenario."""

    twist_turns = simulator.twist_accumulation_turns(result)
    mean_alignment_deg = simulator.mean_body_axis_alignment_deg(result)
    peak_twist_rate_deg_s = float(np.max(np.abs(np.rad2deg(result.twist_rotation_speed))))
    peak_trunk_torque_nm = float(np.max(np.abs(result.tau[:, 6:9])))
    backward_distance_m = abs(parameters.backward_horizontal_velocity) * result.flight_time
    return {
        "label": label,
        "flight_time_s": float(result.flight_time),
        "takeoff_vertical_velocity_m_s": float(parameters.takeoff_vertical_velocity),
        "backward_velocity_m_s": float(parameters.backward_horizontal_velocity),
        "backward_distance_m": float(backward_distance_m),
        "initial_alignment_deg": float(result.initial_body_axis_alignment_deg),
        "mean_alignment_deg": mean_alignment_deg,
        "twist_turns": twist_turns,
        "peak_twist_rate_deg_s": peak_twist_rate_deg_s,
        "peak_trunk_torque_nm": peak_trunk_torque_nm,
        "sigma_global_nms": [float(value) for value in result.equivalent_angular_momentum],
    }


def _save_run_outputs(
    simulator: SkaterFlightSimulator,
    parameters: FlightSimulationParameters,
    result,
    output_dir: Path,
    *,
    label: str | None,
) -> dict[str, Any]:
    """Save one simulation result to disk and return its summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _result_summary(simulator, parameters, result, label=label)
    (output_dir / "parameters.json").write_text(
        json.dumps(_to_serializable(parameters), indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(_to_serializable(summary), indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "timeseries.npz",
        time=result.time,
        q=result.q,
        qdot=result.qdot,
        tau=result.tau,
        angular_momentum=result.angular_momentum,
        body_axis=result.body_axis,
        body_axis_alignment_deg=result.body_axis_alignment_deg,
        twist_rotation_speed=result.twist_rotation_speed,
        twist_angle=result.twist_angle,
        twist_inertia_proxy=result.twist_inertia_proxy,
        markers=result.markers,
        center_of_mass=result.center_of_mass,
        equivalent_angular_momentum=result.equivalent_angular_momentum,
    )
    return summary


def _safe_run_name(index: int, label: str | None) -> str:
    """Return a filesystem-friendly run folder name."""

    if not label:
        return f"run_{index:03d}"
    filtered = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in label
    )
    return f"run_{index:03d}_{filtered.strip('_') or 'scenario'}"


def run_single_simulation(arguments: argparse.Namespace) -> int:
    """Execute and save one command-line simulation."""

    simulator = SkaterFlightSimulator()
    parameters = _parameters_from_arguments(arguments)
    result = simulator.simulate(parameters)
    output_dir = Path(arguments.output_dir)
    summary = _save_run_outputs(simulator, parameters, result, output_dir, label=arguments.label)
    if arguments.print_summary:
        print(json.dumps(summary, indent=2))
    else:
        print(output_dir)
    return 0


def run_batch_simulations(arguments: argparse.Namespace) -> int:
    """Execute a batch of simulations defined in a JSON configuration file."""

    simulator = SkaterFlightSimulator()
    config_path = Path(arguments.config)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "scenarios" not in payload:
        raise ValueError("The batch configuration must be a JSON object with a 'scenarios' list.")
    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("The 'scenarios' entry must be a non-empty list.")

    output_root = Path(arguments.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise ValueError("Each scenario must be a JSON object.")
        label = scenario.get("label")
        parameter_payload = {key: value for key, value in scenario.items() if key != "label"}
        parameters = _parameters_from_payload(parameter_payload)
        result = simulator.simulate(parameters)
        run_dir = output_root / _safe_run_name(index, label)
        summary = _save_run_outputs(simulator, parameters, result, run_dir, label=label)
        summary["run_directory"] = str(run_dir)
        summaries.append(summary)

    (output_root / "batch_summary.json").write_text(
        json.dumps(_to_serializable(summaries), indent=2),
        encoding="utf-8",
    )
    with (output_root / "batch_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(output_root)
    return 0


def export_run_plots(arguments: argparse.Namespace) -> int:
    """Generate standard laboratory figures from one saved run directory."""

    run_dir = Path(arguments.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    timeseries = np.load(run_dir / "timeseries.npz")
    time = np.asarray(timeseries["time"], dtype=float)
    q = np.asarray(timeseries["q"], dtype=float)
    tau = np.asarray(timeseries["tau"], dtype=float)
    body_axis_alignment_deg = np.asarray(timeseries["body_axis_alignment_deg"], dtype=float)
    twist_angle_deg = np.rad2deg(np.asarray(timeseries["twist_angle"], dtype=float))
    twist_rotation_speed_deg_s = np.rad2deg(
        np.asarray(timeseries["twist_rotation_speed"], dtype=float)
    )
    twist_inertia_proxy = np.asarray(timeseries["twist_inertia_proxy"], dtype=float)
    trunk_angles_deg = np.rad2deg(q[:, 6:9])
    trunk_torques = tau[:, 6:9]
    salto_deg = np.rad2deg(q[:, 3])

    figure, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    figure.suptitle(
        f"Run: {summary.get('label') or run_dir.name} | vrille {summary['twist_turns']:.2f} tours"
    )
    axes[0].plot(time, body_axis_alignment_deg, color="#D62728", linewidth=2.0)
    axes[0].set_ylabel("deg")
    axes[0].set_title("Alignement σ / axe longitudinal")

    axes[1].plot(time, twist_angle_deg, color="#FF7F0E", linewidth=2.0, label="Vrille")
    axes[1].plot(time, salto_deg, color="#1F77B4", linewidth=2.0, label="Salto")
    axes[1].legend()
    axes[1].set_ylabel("deg")
    axes[1].set_title("Vrille et salto")

    for axis_index, label in enumerate(("Tronc x", "Tronc y", "Tronc z")):
        axes[2].plot(time, trunk_angles_deg[:, axis_index], linewidth=2.0, label=label)
    axes[2].legend()
    axes[2].set_ylabel("deg")
    axes[2].set_title("3 DoF du tronc")

    for axis_index, label in enumerate(("Couple x", "Couple y", "Couple z")):
        axes[3].plot(time, trunk_torques[:, axis_index], linewidth=2.0, label=label)
    axes[3].legend()
    axes[3].set_ylabel("N.m")
    axes[3].set_title("Efforts du tronc")

    axes[4].plot(time, twist_inertia_proxy, color="#2CA02C", linewidth=2.0, label="||σ|| / |ω|")
    axes[4].plot(time, twist_rotation_speed_deg_s, color="#8C564B", linewidth=2.0, label="ω")
    axes[4].legend()
    axes[4].set_ylabel("mixte")
    axes[4].set_xlabel("Temps (s)")
    axes[4].set_title("Inertie apparente et vitesse de vrille")

    for axis in axes:
        axis.grid(True, alpha=0.25)

    output_path = run_dir / "plots.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(output_path)
    return 0


def compare_batch_results(arguments: argparse.Namespace) -> int:
    """Create a comparison report and a plot from one saved batch directory."""

    batch_dir = Path(arguments.batch_dir)
    batch_summary_path = batch_dir / "batch_summary.json"
    summaries = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("The batch summary must be a non-empty list.")

    metric = arguments.metric
    reverse = not arguments.ascending
    ordered = sorted(summaries, key=lambda item: float(item[metric]), reverse=reverse)

    comparison_json = batch_dir / f"comparison_{metric}.json"
    comparison_csv = batch_dir / f"comparison_{metric}.csv"
    comparison_png = batch_dir / f"comparison_{metric}.png"

    comparison_json.write_text(json.dumps(_to_serializable(ordered), indent=2), encoding="utf-8")
    with comparison_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)

    labels = [item.get("label") or Path(item["run_directory"]).name for item in ordered]
    values = [float(item[metric]) for item in ordered]
    figure, axis = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 5))
    axis.bar(range(len(labels)), values, color="#2E6FBB")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(f"Comparaison batch selon {metric}")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(comparison_png, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(comparison_json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level command-line parser."""

    parser = argparse.ArgumentParser(prog="skating-aerial-alignment")
    subparsers = parser.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser("gui", help="Launch the interactive GUI.")
    gui_parser.set_defaults(handler=lambda _arguments: (launch_app(), 0)[1])

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run one simulation from the command line and save the results.",
    )
    simulate_parser.add_argument("--label", type=str, default=None, help="Optional scenario label.")
    simulate_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder where parameters, summary, and time series will be saved.",
    )
    simulate_parser.add_argument(
        "--sigma-rps",
        type=float,
        nargs=3,
        metavar=("SIGMA_X", "SIGMA_Y", "SIGMA_Z"),
        default=(0.0, 0.0, 3.0),
        help="Equivalent global angular-momentum controls, expressed in rotations/s.",
    )
    simulate_parser.add_argument(
        "--vertical-velocity",
        type=float,
        default=FlightSimulationParameters().takeoff_vertical_velocity,
        help="Take-off vertical CoM velocity in m/s.",
    )
    simulate_parser.add_argument(
        "--backward-velocity",
        type=float,
        default=0.0,
        help="Backward CoM velocity in m/s.",
    )
    simulate_parser.add_argument(
        "--somersault-tilt-deg",
        type=float,
        default=0.0,
        help="Initial somersault tilt in degrees.",
    )
    simulate_parser.add_argument(
        "--inward-tilt-deg",
        type=float,
        default=0.0,
        help="Initial inward tilt in degrees.",
    )
    simulate_parser.add_argument(
        "--initial-trunk-angles-deg",
        type=float,
        nargs=3,
        metavar=("TRUNK_X", "TRUNK_Y", "TRUNK_Z"),
        default=(0.0, 0.0, 0.0),
        help="Initial pelvis-thorax angles in degrees.",
    )
    simulate_parser.add_argument(
        "--initial-trunk-velocity-deg-s",
        type=float,
        nargs=3,
        metavar=("TRUNK_DX", "TRUNK_DY", "TRUNK_DZ"),
        default=(0.0, 0.0, 0.0),
        help="Initial pelvis-thorax angular velocities in degrees/s.",
    )
    simulate_parser.add_argument(
        "--stabilize-trunk",
        action="store_true",
        help="Enable the trunk PD controller.",
    )
    simulate_parser.add_argument(
        "--kp",
        type=float,
        nargs=3,
        metavar=("KP_X", "KP_Y", "KP_Z"),
        default=PDControllerConfiguration().proportional_gains,
        help="Trunk proportional gains.",
    )
    simulate_parser.add_argument(
        "--kd",
        type=float,
        nargs=3,
        metavar=("KD_X", "KD_Y", "KD_Z"),
        default=PDControllerConfiguration().derivative_gains,
        help="Trunk derivative gains.",
    )
    simulate_parser.add_argument(
        "--torque-limits",
        type=float,
        nargs=3,
        metavar=("TAU_X", "TAU_Y", "TAU_Z"),
        default=PDControllerConfiguration().torque_limits,
        help="Torque saturation limits for the trunk controller.",
    )
    simulate_parser.add_argument(
        "--sample-count",
        type=int,
        default=FlightSimulationParameters().sample_count,
        help="Number of time samples retained in the saved output.",
    )
    simulate_parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the result summary to stdout instead of only printing the output path.",
    )
    simulate_parser.set_defaults(handler=run_single_simulation)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run several simulations from a JSON configuration file.",
    )
    batch_parser.add_argument("--config", type=str, required=True, help="Batch JSON file.")
    batch_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root folder where the batch results will be saved.",
    )
    batch_parser.set_defaults(handler=run_batch_simulations)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare and rank the runs inside a saved batch directory.",
    )
    compare_parser.add_argument(
        "--batch-dir",
        type=str,
        required=True,
        help="Batch result directory containing batch_summary.json.",
    )
    compare_parser.add_argument(
        "--metric",
        type=str,
        default="twist_turns",
        choices=(
            "twist_turns",
            "mean_alignment_deg",
            "initial_alignment_deg",
            "peak_twist_rate_deg_s",
            "peak_trunk_torque_nm",
            "flight_time_s",
            "backward_distance_m",
        ),
        help="Metric used to rank the runs.",
    )
    compare_parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort the selected metric in ascending order instead of descending.",
    )
    compare_parser.set_defaults(handler=compare_batch_results)

    export_parser = subparsers.add_parser(
        "export-plots",
        help="Generate standard figures from one saved run directory.",
    )
    export_parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing summary.json and timeseries.npz.",
    )
    export_parser.set_defaults(handler=export_run_plots)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI entry point."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        arguments = parser.parse_args(["gui"])
    return int(arguments.handler(arguments))
