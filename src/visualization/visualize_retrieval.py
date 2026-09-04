import argparse
import html
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np


def haversine_km(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> Optional[float]:
    """Computes the great-circle distance in kilometers between two points."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    try:
        lat1_f, lon1_f, lat2_f, lon2_f = (
            float(lat1),
            float(lon1),
            float(lat2),
            float(lon2),
        )
        if np.isnan(lat1_f) or np.isnan(lon1_f) or np.isnan(lat2_f) or np.isnan(lon2_f):
            return None
        lat1_r, lon1_r, lat2_r, lon2_r = map(
            np.radians, [lat1_f, lon1_f, lat2_f, lon2_f]
        )
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return round(float(6371.0 * c), 1)
    except Exception:
        return None


def select_balanced_samples(
    queries: List[Dict[str, Any]], max_samples: int = 100, seed: int = 42
) -> List[Dict[str, Any]]:
    """Selects a representative sample of queries balancing classes and match/mismatch outcomes."""
    if max_samples <= 0 or len(queries) <= max_samples:
        return queries

    rng = random.Random(seed)
    class_outcomes: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for q in queries:
        cls = str(q.get("ground_truth", "Unknown"))
        is_correct = bool(q.get("p1", 0) >= 1.0)
        bucket = "correct" if is_correct else "failed"
        if cls not in class_outcomes:
            class_outcomes[cls] = {"correct": [], "failed": []}
        class_outcomes[cls][bucket].append(q)

    # Calculate target per class
    classes = sorted(list(class_outcomes.keys()))
    if not classes:
        return queries[:max_samples]

    samples_per_class = max(1, max_samples // len(classes))
    selected: List[Dict[str, Any]] = []

    for cls in classes:
        correct_pool = class_outcomes[cls]["correct"]
        failed_pool = class_outcomes[cls]["failed"]
        rng.shuffle(correct_pool)
        rng.shuffle(failed_pool)

        half = max(1, samples_per_class // 2)
        take_correct = correct_pool[:half]
        take_failed = failed_pool[:half]
        selected.extend(take_correct)
        selected.extend(take_failed)

        # Fill remaining if one pool had fewer
        rem = samples_per_class - len(take_correct) - len(take_failed)
        if rem > 0:
            extra = (correct_pool[half:] + failed_pool[half:])[:rem]
            selected.extend(extra)

    # If still under max_samples, sample from remaining queries
    if len(selected) < max_samples:
        selected_ids = {id(item) for item in selected}
        remaining = [q for q in queries if id(q) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: (max_samples - len(selected))])

    # If over max_samples, truncate
    if len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]

    return selected


def load_retrieval_dashboard_template() -> str:
    """Loads the HTML template for the retrieval dashboard from templates/retrieval_dashboard.html."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(repo_root, "templates", "retrieval_dashboard.html"),
        os.path.join(
            os.path.dirname(__file__), "templates", "retrieval_dashboard.html"
        ),
        "templates/retrieval_dashboard.html",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(
        f"Could not find retrieval_dashboard.html in candidate paths: {candidates}"
    )


def generate_retrieval_html(
    output_html_path: str,
    benchmark_title: str,
    model_name: str,
    summary_metrics: Dict[str, Any],
    queries_data: List[Dict[str, Any]],
    max_samples: int = 100,
) -> str:
    """
    Renders an interactive HTML dashboard showing queries alongside their top-K retrieved matches.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)

    selected_queries = select_balanced_samples(queries_data, max_samples=max_samples)

    # Format metrics for display
    p1 = f"{float(summary_metrics.get('p@1', 0.0)):.1f}"
    p5 = f"{float(summary_metrics.get('p@5', 0.0)):.1f}"
    p10 = f"{float(summary_metrics.get('p@10', 0.0)):.1f}"
    map10 = f"{float(summary_metrics.get('map@10', 0.0)):.1f}"
    mrr10 = f"{float(summary_metrics.get('mrr@10', 0.0)):.1f}"

    html_template = load_retrieval_dashboard_template()
    html_content = (
        html_template.replace("{{TITLE}}", html.escape(benchmark_title))
        .replace("{{MODEL_NAME}}", html.escape(model_name))
        .replace("{{TOTAL_QUERIES}}", f"{len(queries_data):,}")
        .replace("{{SHOWN_QUERIES}}", f"{len(selected_queries):,}")
        .replace("{{P1}}", p1)
        .replace("{{P5}}", p5)
        .replace("{{P10}}", p10)
        .replace("{{MAP10}}", map10)
        .replace("{{MRR10}}", mrr10)
        .replace("{{QUERY_DATA_JSON}}", json.dumps(selected_queries))
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(
        f" -> Interactive retrieval visualizer saved to: {os.path.abspath(output_html_path)}"
    )
    return output_html_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML retrieval visualizer from benchmark JSON/CSV."
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing queries and retrieved items.",
    )
    parser.add_argument(
        "--output_html", type=str, required=True, help="Path to output HTML file."
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Spatial-Semantic Representation Retrieval Visualizer",
        help="Title for the dashboard.",
    )
    parser.add_argument(
        "--model_name", type=str, default="Vision Model", help="Model name."
    )
    parser.add_argument(
        "--max_samples", type=int, default=100, help="Max queries to visualize."
    )
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    queries = data.get("queries", [])

    generate_retrieval_html(
        args.output_html,
        args.title,
        args.model_name,
        metrics,
        queries,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
