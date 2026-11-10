"""Run validation experiments or redraw figures from saved results.

Example: python experiments/run.py all
         python experiments/run.py noise_sampling --plot-only
         python experiments/run.py solo_idealised
         python experiments/run.py idealised --plot-only

The 'all' selector runs both supplementary sensitivity experiments.
The 'idealised' selector runs SOLO, DOPPIO, LATTE and the schematic.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.sensitivity import (Config, load_config, prepare_experiment, run_self_tests,
                                      run_noise_sampling, run_background_flow)
from experiments.outputs import (save_results, load_saved, write_report, save_plot_metadata)


def render(results, context, directory):
    # Imported only here: plot-only work never invokes fitting or self-tests.
    import matplotlib.pyplot as plt
    from experiments.plotting import plot_noise_sampling, plot_background_flow, plot_noise_geometry, save_figure
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    name = results.attrs["experiment"]
    plotting = plot_noise_sampling if name == "noise_sampling" else plot_background_flow
    figure = plotting(results, context.config)
    paths = list(save_figure(figure, directory / "figures" / name, context.config).values())
    plt.close(figure)
    if name == "noise_sampling":
        geometry = plot_noise_geometry(context)
        paths += list(save_figure(geometry, directory / "figures" / "sampling_geometry", context.config).values())
        plt.close(geometry)
    write_report(results, context, directory)
    save_plot_metadata(directory, context, paths)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["all", "noise_sampling", "background_flow", "solo_idealised",
                                               "doppio_idealised", "latte_idealised", "schematic", "idealised"])
    parser.add_argument("--config", type=Path, help="Defaults to the selected experiment's configuration")
    parser.add_argument("--output-root", type=Path, help="Override results root; relative paths are repository-relative")
    parser.add_argument("--replicates", type=int, help="Override noise replicate count, e.g. 2 for a smoke test")
    parser.add_argument("--plot-only", action="store_true", help="Redraw saved trials; never run fitting")
    args = parser.parse_args(argv)
    if args.replicates is not None and (args.replicates < 1 or args.plot_only):
        parser.error("--replicates must be positive and cannot be used with --plot-only")
    if args.experiment == "idealised":
        if args.config is not None or args.replicates is not None:
            parser.error("idealised uses each group's own default config; select one group for --config")
        from experiments import solo, idealised
        directories = []
        for name in ("solo_idealised", *idealised.NAMES):
            command = [name]
            if args.plot_only:
                command.append("--plot-only")
            if args.output_root is not None:
                command += ["--output-root",str(args.output_root)]
            main(command)
            config = solo.load_config() if name == "solo_idealised" else idealised.load_config(ROOT/f"experiments/configs/{name}.json",name)
            directories.append(idealised.output_directory(config,args.output_root))
        summary = directories[0].parent/"idealised_manuscript_summary.md"
        summary.write_text("# Idealised experiments — combined manuscript results\n\n"+
            "\n\n---\n\n".join((d/"manuscript_summary.md").read_text() for d in directories))
        print(f"Combined manuscript summary: {summary}",flush=True)
        return 0
    if args.experiment in ("doppio_idealised", "latte_idealised", "schematic"):
        if args.replicates is not None:
            parser.error("These cases are deterministic; --replicates is inapplicable")
        from experiments import idealised
        from experiments.idealised_plotting import render as render_idealised
        config_path = args.config or ROOT/f"experiments/configs/{args.experiment}.json"
        values = idealised.load_config(config_path,args.experiment)
        directory = idealised.output_directory(values,args.output_root)
        if args.plot_only:
            data,values = idealised.load_saved(directory,values)
        else:
            data = idealised.run_experiment(values)
            directory = idealised.save_results(data,values,config_path,args.output_root)
        render_idealised(data,values,directory)
        print(f"{'Redrawn without fitting' if args.plot_only else 'Saved '+args.experiment}: {directory}",flush=True)
        if not args.plot_only and "parameters" in data:
            print(f"Full fits accepted: {data['parameters'].status.eq('optimized').sum()}/{len(data['parameters'])}",flush=True)
        return 0
    if args.experiment == "solo_idealised":
        if args.replicates is not None:
            parser.error("SOLO idealised cases are deterministic; --replicates is inapplicable")
        from experiments import solo
        from experiments.solo_plotting import render as render_solo
        config_path = args.config or solo.DEFAULT_CONFIG
        values = solo.load_config(config_path)
        if args.plot_only:
            root = Path(args.output_root or values["output_directory"])
            directory = (root if root.is_absolute() else ROOT/root)/"solo_idealised"
            data, values = solo.load_saved(directory, values)
        else:
            data = solo.run_experiment(values)
            directory = solo.save_results(data, values, config_path, args.output_root)
        render_solo(data, values, directory)
        print(f"{'Redrawn without fitting' if args.plot_only else 'Saved SOLO idealised results'}: {directory}", flush=True)
        if not args.plot_only:
            print(f"Outer fits accepted: {(data['parameters'].status == 'optimized').sum()}/{len(data['parameters'])}", flush=True)
        return 0
    config = load_config(args.config or ROOT / "experiments/configs/sensitivity.json")
    values = deepcopy(config.values);overrides = {}
    if args.output_root is not None:
        values["output_directory"] = str(args.output_root);overrides["output_root"] = str(args.output_root)
    if args.replicates is not None:
        values["noise"]["replicates"] = args.replicates;overrides["replicates"] = args.replicates
    config = Config(values, config.path)
    names = ["noise_sampling", "background_flow"] if args.experiment == "all" else [args.experiment]
    if args.plot_only:
        for name in names:
            directory = config.output / name
            results, context = load_saved(directory, config)
            render(results, context, directory)
            print(f"Redrawn without fitting: {directory}", flush=True)
        return 0
    context = prepare_experiment(config)
    checks = run_self_tests(context)
    print(f"Self-tests passed. Uref={context.truth['u_ref_m_s']:.9g} m/s; all distances/fits use SI.", flush=True)
    for name in names:
        results = (run_noise_sampling if name == "noise_sampling" else run_background_flow)(context, checks)
        directory = save_results(results, context, overrides)
        render(results, context, directory)
        print(f"Saved {name}: {int(results.success.sum())}/{len(results)} accepted; {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
