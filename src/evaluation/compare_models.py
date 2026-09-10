import argparse
import datetime
import glob
import os
import re
import subprocess
import sys


def find_visualizers_for_model(actual_html_path):
    """
    Finds all generated HTML visualizers for a given model run,
    including per-representation files (_cls.html, _avg_patch.html, etc.).
    Returns a list of tuples: (display_label, file_path).
    """
    if not actual_html_path:
        return []

    html_dir = os.path.dirname(os.path.abspath(actual_html_path))
    html_base = os.path.splitext(os.path.basename(actual_html_path))[0]
    html_ext = os.path.splitext(actual_html_path)[1]

    pattern = os.path.join(html_dir, f"{html_base}*{html_ext}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return []

    slug_labels = [
        ("_1st_cls_fg_removed", "1st CLS + FG-Removed"),
        ("_2nd_cls_fg_removed", "2nd CLS + FG-Removed"),
        ("_cls_fg_removed", "CLS + FG-Removed"),
        ("_fg_removed", "FG-Removed Average"),
        ("_1st_cls_avg_patch", "1st CLS + Avg Patch"),
        ("_2nd_cls_avg_patch", "2nd CLS + Avg Patch"),
        ("_cls_avg_patch", "CLS + Avg Patch"),
        ("_1st_cls", "1st CLS"),
        ("_2nd_cls", "2nd CLS"),
        ("_cls", "CLS"),
        ("_avg_patch", "Average Patch"),
        ("_seg_masked", "Seg-Masked"),
    ]

    found = []
    has_specific_reps = False
    for f in matches:
        for suffix, label in slug_labels:
            if f.endswith(f"{suffix}{html_ext}"):
                found.append((label, f))
                has_specific_reps = True
                break

    if not has_specific_reps:
        for f in matches:
            if f == actual_html_path:
                found.append(("Dashboard", f))

    return found


BENCHMARKS = {
    "lucas": {
        "module": "src.evaluation.benchmark_lucas",
        "yaml_key": "lucas",
        "yaml_file": "config/evaluation/params_offline.yaml",
        "report_prefix": "lucas_report",
        "extra_args": [
            ("--csv", "csv"),
            ("--img_dir", "img_dir"),
            ("--eunis_raster", "eunis_raster"),
            ("--env_zones_raster", "env_zones_raster"),
        ],
    },
    "places": {
        "module": "src.evaluation.benchmark_places",
        "yaml_key": "places",
        "yaml_file": "config/evaluation/params_offline.yaml",
        "report_prefix": "places_report",
        "extra_args": [("--labels", "labels"), ("--img_dir", "img_dir")],
    },
    "eunis": {
        "module": "src.evaluation.benchmark_eunis",
        "yaml_key": "eunis",
        "yaml_file": "config/evaluation/params_online.yaml",
        "report_prefix": "eunis_report",
        "extra_args": [
            ("--csv_path", "csv_path"),
            ("--raster", "raster"),
            ("--offline_dataset_dirs", "offline_dataset_dirs"),
            ("--query_platform", "query_platform"),
        ],
    },
    "env_zones": {
        "module": "src.evaluation.benchmark_environmental_zones",
        "yaml_key": "environmental_zones",
        "yaml_file": "config/evaluation/params_online.yaml",
        "report_prefix": "environmental_zones_report",
        "extra_args": [
            ("--csv_path", "csv_path"),
            ("--raster", "raster"),
            ("--offline_dataset_dirs", "offline_dataset_dirs"),
            ("--query_platform", "query_platform"),
        ],
    },
}


def parse_report_file(report_path, model_name):
    """Parses a generated text report file to extract representation metrics."""
    results = []
    current_evaluation = "Default"

    if not os.path.exists(report_path):
        print(f"Warning: Report file not found: {report_path}")
        return results

    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Detect section evaluation type
            match_eval = re.search(r"--- (.*?) Evaluation ---", line)
            if match_eval:
                current_evaluation = match_eval.group(1).strip()
                continue

            # Parse table rows containing metric data
            if (
                "|" in line
                and not line.startswith("Representation")
                and "---" not in line
                and "===" not in line
            ):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6:
                    rep = parts[0]
                    # Strip model name or clean version from representation name
                    model_clean = model_name.replace("/", "_")
                    model_base = model_name.split("/")[-1]
                    for prefix in [model_name, model_clean, model_base]:
                        if rep.startswith(prefix):
                            rep = rep[len(prefix) :].strip()
                    # Strip leading spaces/hyphens/underscores
                    rep = re.sub(r"^[-_\s]+", "", rep)

                    # Extract precision suffix if present
                    precision = "FP32"  # default
                    match_prec = re.search(r"\s+\((FP16|FP32)\)$", rep)
                    if match_prec:
                        precision = match_prec.group(1)
                        rep = rep[: -len(match_prec.group(0))].strip()

                    results.append(
                        {
                            "Model": model_name,
                            "Evaluation": current_evaluation,
                            "Representation": rep,
                            "Precision": precision,
                            "P@1": parts[1],
                            "P@5": parts[2],
                            "P@10": parts[3],
                            "MAP@10": parts[4],
                            "MRR@10": parts[5],
                        }
                    )
    return results


def deduplicate_cnn_rows(results):
    """
    If a model doesn't have a real CLS token, CLS and other representation
    combinations (like CLS + Avg Patch) will have identical scores to
    Average Patch. We filter out the redundant rows for such cases.
    """
    grouped = {}
    for r in results:
        key = (r["Model"], r["Evaluation"], r["Precision"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    filtered_results = []
    for key, rows in grouped.items():
        avg_row = next(
            (r for r in rows if r["Representation"] == "Average Patch"), None
        )

        if avg_row:
            cleaned_rows = [avg_row]

            for r in rows:
                if r is avg_row:
                    continue
                if r["Representation"] in [
                    "CLS",
                    "CLS + Avg Patch",
                    "FG-Removed Average",
                    "CLS + FG-Removed Avg",
                    "CLS + FG-Removed Average",
                ]:
                    metrics_match = (
                        r["P@1"] == avg_row["P@1"]
                        and r["P@5"] == avg_row["P@5"]
                        and r["P@10"] == avg_row["P@10"]
                        and r["MAP@10"] == avg_row["MAP@10"]
                        and r["MRR@10"] == avg_row["MRR@10"]
                    )
                    if metrics_match:
                        continue
                cleaned_rows.append(r)

            if len(cleaned_rows) < len(rows):
                avg_row["Representation"] = "Average (No CLS)"

            rows = cleaned_rows

        filtered_results.extend(rows)
    return filtered_results


def main():
    parser = argparse.ArgumentParser(
        description="Collate and compare multiple model representations on a specific dataset."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARKS.keys()),
        help="The target benchmark script to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of model names/identifiers to compare.",
    )
    args = parser.parse_args()

    bench_config = BENCHMARKS[args.benchmark]

    # Load configuration parameters from YAML (with local overrides)
    yaml_path = bench_config["yaml_file"]
    from src.utils.config import load_config

    config_data = load_config(yaml_path)
    if not config_data:
        print(f"Error: YAML configuration file not found at: {yaml_path}")
        sys.exit(1)

    params = config_data.get(bench_config["yaml_key"], {})

    print("=" * 90)
    print(f"Starting Collated Model Comparison for benchmark: {args.benchmark.upper()}")
    print(f"Config loaded from: {yaml_path}")
    print(f"Models to evaluate: {args.models}")
    print("=" * 90)

    # Source local credentials from .env to environment if available
    env = os.environ.copy()
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()

    all_results = []
    visualizer_links = []

    # Run each model sequentially
    for idx, model in enumerate(args.models):
        print(f"\n[{idx+1}/{len(args.models)}] Evaluating model: {model}...")

        # Prepare execution commands
        model_clean = model.replace("/", "_")
        report_path = (
            f"./benchmark_results/{bench_config['report_prefix']}_{model_clean}.txt"
        )
        csv_path = f"./benchmark_results/{bench_config['report_prefix'].replace('report', 'results')}_{model_clean}.csv"
        html_path = f"./benchmark_results/{bench_config['report_prefix'].replace('report', 'visualizer')}_{model_clean}.html"

        # Check if the outputs with the dynamic parameters suffix already exist to skip execution
        seed = str(params.get("seed", 42))
        num_queries = str(params.get("num_queries", 3000))
        plat_val = str(params.get("query_platform", "")).lower().strip()
        plat_suffix = ""
        if args.benchmark in ["env_zones", "eunis"]:
            plat_suffix = (
                f"_plat-{plat_val}"
                if plat_val and plat_val not in ["none", "null"]
                else "_plat-all"
            )
        suffix = f"_s{seed}_q{num_queries}{plat_suffix}"
        actual_report_path = report_path.replace(".txt", f"{suffix}.txt")
        actual_csv_path = csv_path.replace(".csv", f"{suffix}.csv")
        actual_html_path = html_path.replace(".html", f"{suffix}.html")

        if os.path.exists(actual_report_path) and os.path.exists(actual_csv_path):
            print(
                f" -> Existing report found: {actual_report_path}. Skipping evaluation run."
            )
            model_results = parse_report_file(actual_report_path, model)
            all_results.extend(model_results)
            viz_files = find_visualizers_for_model(actual_html_path)
            if viz_files:
                visualizer_links.append((model, viz_files))
            continue

        cmd = [
            "python3",
            "-m",
            bench_config["module"],
            "--model_name",
            model,
            "--num_queries",
            str(params.get("num_queries", 3000)),
            "--num_database",
            str(params.get("num_database", 0)),
            "--batch_size",
            str(params.get("batch_size", 32)),
            "--seed",
            str(params.get("seed", 42)),
            "--output_report",
            report_path,
            "--output_csv",
            csv_path,
        ]

        # Add HTML visualization flag if enabled in yaml and supported
        viz_samples = params.get("visualize_samples")
        if (
            viz_samples
            and viz_samples not in [0, "0", "false", False]
            and args.benchmark in ["env_zones", "eunis"]
        ):
            cmd.extend(
                [
                    "--output_html",
                    html_path,
                    "--visualize_samples",
                    str(viz_samples),
                ]
            )

        # Add benchmark-specific path arguments
        for cli_flag, yaml_name in bench_config["extra_args"]:
            if yaml_name in params:
                cmd.extend([cli_flag, str(params[yaml_name])])

        # Optional flags
        if params.get("use_segformer") is False:
            cmd.append("--no_segformer")
        if params.get("compare_clip") is True:
            cmd.append("--compare_clip")
        if params.get("use_fg_removal") is False:
            cmd.append("--no_fg_removal")
        if "fg_attn_threshold" in params and params["fg_attn_threshold"] is not None:
            cmd.extend(["--fg_attn_threshold", str(params["fg_attn_threshold"])])
        if "max_fg_ratio" in params and params["max_fg_ratio"] is not None:
            cmd.extend(["--max_fg_ratio", str(params["max_fg_ratio"])])
        if "discard_classes" in params and params["discard_classes"] is not None:
            cmd.append("--discard_classes")
            cmd.extend([str(c) for c in params["discard_classes"]])

        # Run benchmark
        try:
            subprocess.run(cmd, env=env, check=True)
            viz_files = find_visualizers_for_model(actual_html_path)
            if viz_files:
                visualizer_links.append((model, viz_files))
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for model {model}: {e}")
            continue

        # Parse metrics
        model_results = parse_report_file(actual_report_path, model)
        all_results.extend(model_results)

    if not all_results:
        print("\nError: No benchmarking metrics were successfully parsed.")
        sys.exit(1)

    # Deduplicate redundant rows for CNN/No-CLS models
    all_results = deduplicate_cnn_rows(all_results)

    # Generate comparative Markdown table
    collate_path = f"./benchmark_results/comparison_{args.benchmark}.md"
    os.makedirs(os.path.dirname(collate_path), exist_ok=True)

    def parse_pct(pct_str):
        try:
            return float(pct_str.replace("%", "").strip())
        except Exception:
            return -1.0

    with open(collate_path, "w", encoding="utf-8") as f:
        f.write(f"# Comparative Benchmark Report: {args.benchmark.upper()}\n")
        f.write(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Group results by Evaluation category
        categories = sorted(list(set(r["Evaluation"] for r in all_results)))
        for category in categories:
            f.write(f"## {category} Comparison\n\n")
            f.write(
                "| Model | Representation | Precision | P@1 | P@5 | P@10 | MAP@10 | MRR@10 |\n"
            )
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

            cat_results = [r for r in all_results if r["Evaluation"] == category]
            # Sort by P@1 metric descending
            cat_results = sorted(
                cat_results, key=lambda x: parse_pct(x["P@1"]), reverse=True
            )

            for r in cat_results:
                f.write(
                    f"| {r['Model']} | {r['Representation']} | {r['Precision']} | {r['P@1']} | {r['P@5']} | {r['P@10']} | {r['MAP@10']} | {r['MRR@10']} |\n"
                )
            f.write("\n")

        if visualizer_links:
            f.write("## Interactive Retrieval Visualizers\n\n")
            f.write("| Model | Representation | Visualizer Dashboard |\n")
            f.write("| :--- | :--- | :--- |\n")
            for model_name, v_files in visualizer_links:
                for rep_label, v_path in v_files:
                    rel_path = os.path.basename(v_path)
                    f.write(
                        f"| {model_name} | {rep_label} | [{rel_path}]({rel_path}) |\n"
                    )
            f.write("\n")

    print("\n" + "=" * 90)
    print("🎉 Collated Model Comparison completed successfully!")
    print(f"Results saved to: {os.path.abspath(collate_path)}")
    print("=" * 90)

    # Print summary table to console
    with open(collate_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
