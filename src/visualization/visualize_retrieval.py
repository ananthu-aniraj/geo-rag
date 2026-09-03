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


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --bg-subtle: #f1f5f9;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --primary: #0f766e;
            --primary-hover: #115e59;
            --success: #16a34a;
            --success-bg: #dcfce7;
            --danger: #dc2626;
            --danger-bg: #fee2e2;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.08);
            --font-mono: 'JetBrains Mono', monospace;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-page: #0f172a;
                --bg-card: #1e293b;
                --bg-subtle: #334155;
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --text-muted: #94a3b8;
                --border-color: #334155;
                --primary: #14b8a6;
                --primary-hover: #2dd4bf;
                --success: #4ade80;
                --success-bg: #064e3b;
                --danger: #f87171;
                --danger-bg: #7f1d1d;
                --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
                --shadow-md: 0 4px 8px rgba(0,0,0,0.4);
            }
        }

        * { box-sizing: border-box; }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-page);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }

        .container { max-width: 1440px; margin: 0 auto; }

        .header {
            background: linear-gradient(135deg, #134e4a 0%, #0f766e 100%);
            color: white;
            padding: 32px 36px;
            border-radius: 16px;
            box-shadow: var(--shadow-md);
            margin-bottom: 24px;
        }
        .header h1 { margin: 0 0 8px 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
        .header p { margin: 0; opacity: 0.9; font-size: 0.95rem; }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-top: 24px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 16px;
        }
        .metric-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.85; }
        .metric-value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; font-family: var(--font-mono); }

        .controls-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-sm);
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
            justify-content: space-between;
        }

        .filter-group { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
        .btn-filter {
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: var(--bg-subtle);
            color: var(--text-primary);
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-filter:hover { border-color: var(--primary); }
        .btn-filter.active { background: var(--primary); color: white; border-color: var(--primary); }

        select, input[type="text"] {
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: var(--bg-card);
            color: var(--text-primary);
            font-size: 0.85rem;
            font-family: inherit;
        }

        .prefix-remapper {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
            width: 100%;
            padding-top: 12px;
            border-top: 1px dashed var(--border-color);
        }

        .results-list { display: flex; flex-direction: column; gap: 20px; }

        .query-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            box-shadow: var(--shadow-sm);
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .query-card { grid-template-columns: 1fr; }
        }

        .query-sidebar {
            display: flex;
            flex-direction: column;
            gap: 12px;
            border-right: 1px solid var(--border-color);
            padding-right: 20px;
        }
        @media (max-width: 900px) {
            .query-sidebar { border-right: none; border-bottom: 1px solid var(--border-color); padding-right: 0; padding-bottom: 16px; }
        }

        .thumb-box {
            position: relative;
            width: 100%;
            height: 190px;
            border-radius: 10px;
            overflow: hidden;
            background: var(--bg-subtle);
            border: 1px solid var(--border-color);
            cursor: pointer;
        }
        .thumb-box img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.2s ease;
        }
        .thumb-box:hover img { transform: scale(1.03); }

        .badge-tag {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-query { background: var(--bg-subtle); color: var(--text-secondary); border: 1px solid var(--border-color); }
        .badge-match { background: var(--success-bg); color: var(--success); }
        .badge-mismatch { background: var(--danger-bg); color: var(--danger); }

        .retrieved-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
            gap: 12px;
        }

        .retrieved-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            position: relative;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .retrieved-card.is-match { border-color: var(--success); border-width: 2px; }
        .retrieved-card.is-mismatch { border-color: var(--danger); border-width: 2px; }
        .retrieved-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }

        .retrieved-thumb {
            width: 100%;
            height: 120px;
            border-radius: 6px;
            overflow: hidden;
            background: var(--bg-subtle);
            position: relative;
        }
        .retrieved-thumb img { width: 100%; height: 100%; object-fit: cover; }

        .rank-badge {
            position: absolute;
            top: 6px;
            left: 6px;
            background: rgba(0,0,0,0.75);
            color: white;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: var(--font-mono);
        }

        .retrieved-info {
            font-size: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .retrieved-label {
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .retrieved-meta {
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 0.7rem;
            display: flex;
            justify-content: space-between;
        }

        /* Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0; top: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            backdrop-filter: blur(4px);
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .modal img { max-width: 90vw; max-height: 85vh; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{{TITLE}}</h1>
        <p>Model: <strong>{{MODEL_NAME}}</strong> &bull; Total Evaluated Queries: <strong>{{TOTAL_QUERIES}}</strong> &bull; Exported Diagnostics: <strong>{{SHOWN_QUERIES}}</strong></p>
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">Precision @ 1</div><div class="metric-value">{{P1}}%</div></div>
            <div class="metric-card"><div class="metric-label">Precision @ 5</div><div class="metric-value">{{P5}}%</div></div>
            <div class="metric-card"><div class="metric-label">Precision @ 10</div><div class="metric-value">{{P10}}%</div></div>
            <div class="metric-card"><div class="metric-label">Mean AP @ 10</div><div class="metric-value">{{MAP10}}%</div></div>
            <div class="metric-card"><div class="metric-label">MRR @ 10</div><div class="metric-value">{{MRR10}}%</div></div>
        </div>
    </div>

    <div class="controls-card">
        <div class="filter-group">
            <button class="btn-filter active" onclick="filterOutcome('all', this)">All (<span id="count-all">0</span>)</button>
            <button class="btn-filter" onclick="filterOutcome('correct', this)">🟢 P@1 Match (<span id="count-correct">0</span>)</button>
            <button class="btn-filter" onclick="filterOutcome('failed', this)">🔴 P@1 Mismatch (<span id="count-failed">0</span>)</button>
        </div>

        <div class="filter-group">
            <select id="classSelect" onchange="applyFilters()">
                <option value="all">All Classes / Zones</option>
            </select>
            <input type="text" id="searchInput" placeholder="Search ID, class, platform..." oninput="applyFilters()">
        </div>

        <div class="prefix-remapper">
            <span><strong>Offline Path Remapper:</strong></span>
            <input type="text" id="findPrefix" placeholder="Base path to replace..." style="flex: 1;" oninput="renderQueries()">
            <span>➔</span>
            <input type="text" id="replacePrefix" placeholder="e.g. http://localhost:8000/" style="flex: 1;" oninput="renderQueries()">
        </div>
    </div>

    <div id="resultsContainer" class="results-list"></div>
    <div style="text-align: center; margin: 30px 0;">
        <button id="loadMoreBtn" class="btn-filter" style="padding: 12px 28px; font-size: 1rem;" onclick="loadMore()">Load More Queries</button>
    </div>
</div>

<div id="imgModal" class="modal" onclick="this.style.display='none'">
    <img id="modalImg" src="">
</div>

<script>
    const QUERY_DATA = {{QUERY_DATA_JSON}};
    let currentOutcomeFilter = 'all';
    let filteredData = [...QUERY_DATA];
    let renderLimit = 30;

    function init() {
        // Populate class dropdown
        const classSelect = document.getElementById('classSelect');
        const classes = [...new Set(QUERY_DATA.map(q => q.ground_truth))].filter(Boolean).sort();
        classes.forEach(cls => {
            const opt = document.createElement('option');
            opt.value = cls;
            opt.textContent = cls;
            classSelect.appendChild(opt);
        });

        // Update count badges
        const countCorrect = QUERY_DATA.filter(q => (q.p1 || 0) >= 1.0).length;
        const countFailed = QUERY_DATA.length - countCorrect;
        document.getElementById('count-all').textContent = QUERY_DATA.length;
        document.getElementById('count-correct').textContent = countCorrect;
        document.getElementById('count-failed').textContent = countFailed;

        // Auto-detect local path prefix if any
        for (let q of QUERY_DATA) {
            if (q.query_url && (q.query_url.startsWith('/') || q.query_url.startsWith('file://'))) {
                const idx = q.query_url.lastIndexOf('/');
                if (idx > 0) {
                    document.getElementById('findPrefix').value = q.query_url.substring(0, idx + 1);
                    break;
                }
            }
        }

        applyFilters();
    }

    function filterOutcome(outcome, btn) {
        currentOutcomeFilter = outcome;
        document.querySelectorAll('.controls-card .btn-filter').forEach(b => b.classList.remove('active'));
        if (btn) btn.classList.add('active');
        applyFilters();
    }

    function applyFilters() {
        const selectedClass = document.getElementById('classSelect').value;
        const searchTxt = document.getElementById('searchInput').value.toLowerCase().trim();

        filteredData = QUERY_DATA.filter(q => {
            if (currentOutcomeFilter === 'correct' && (q.p1 || 0) < 1.0) return false;
            if (currentOutcomeFilter === 'failed' && (q.p1 || 0) >= 1.0) return false;
            if (selectedClass !== 'all' && q.ground_truth !== selectedClass) return false;
            if (searchTxt) {
                const qText = `${q.query_id} ${q.ground_truth} ${q.query_platform}`.toLowerCase();
                if (!qText.includes(searchTxt)) return false;
            }
            return true;
        });

        renderLimit = 30;
        renderQueries();
    }

    function resolveUrl(rawUrl) {
        if (!rawUrl) return '';
        const findP = document.getElementById('findPrefix').value;
        const replaceP = document.getElementById('replacePrefix').value;
        if (findP && replaceP && rawUrl.startsWith(findP)) {
            return rawUrl.replace(findP, replaceP);
        }
        return rawUrl;
    }

    function openModal(url) {
        const resolved = resolveUrl(url);
        document.getElementById('modalImg').src = resolved;
        document.getElementById('imgModal').style.display = 'flex';
    }

    function renderQueries() {
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '';

        const toShow = filteredData.slice(0, renderLimit);
        toShow.forEach((q, idx) => {
            const card = document.createElement('div');
            card.className = 'query-card';

            const isTop1Match = (q.p1 || 0) >= 1.0;
            const matchBadgeClass = isTop1Match ? 'badge-match' : 'badge-mismatch';
            const matchBadgeText = isTop1Match ? '✓ Top-1 Match' : '✗ Top-1 Miss';

            const queryThumbUrl = resolveUrl(q.query_url);

            let retrievedHtml = '';
            (q.retrieved || []).forEach(r => {
                const rThumbUrl = resolveUrl(r.url);
                const rMatchClass = r.is_match ? 'is-match' : 'is-mismatch';
                const rBadgeClass = r.is_match ? 'badge-match' : 'badge-mismatch';
                const rBadgeText = r.is_match ? 'MATCH' : 'MISMATCH';

                const distText = (r.distance_km !== null && r.distance_km !== undefined) ? `${r.distance_km} km` : '';

                retrievedHtml += `
                    <div class="retrieved-card ${rMatchClass}">
                        <div class="retrieved-thumb" onclick="openModal('${r.url}')">
                            <span class="rank-badge">#${r.rank}</span>
                            <img src="${rThumbUrl}" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\'><rect fill=\\'%23334155\\' width=\\'100\\' height=\\'100\\'/><text fill=\\'%2394a3b8\\' x=\\'50%\\' y=\\'50%\\' text-anchor=\\'middle\\' font-size=\\'11\\'>No Preview</text></svg>'">
                        </div>
                        <div class="retrieved-info">
                            <span class="badge-tag ${rBadgeClass}" style="align-self: flex-start;">${rBadgeText}</span>
                            <div class="retrieved-label" title="${r.predicted_label}">${r.predicted_label}</div>
                            <div class="retrieved-meta">
                                <span>cos: ${r.similarity || 0}</span>
                                <span>${distText}</span>
                            </div>
                        </div>
                    </div>
                `;
            });

            card.innerHTML = `
                <div class="query-sidebar">
                    <div class="thumb-box" onclick="openModal('${q.query_url}')">
                        <img src="${queryThumbUrl}" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\'><rect fill=\\'%23334155\\' width=\\'100\\' height=\\'100\\'/><text fill=\\'%2394a3b8\\' x=\\'50%\\' y=\\'50%\\' text-anchor=\\'middle\\' font-size=\\'12\\'>Query Image</text></svg>'">
                    </div>
                    <div>
                        <span class="badge-tag badge-query">${q.query_platform || 'query'}</span>
                        <span class="badge-tag ${matchBadgeClass}">${matchBadgeText}</span>
                    </div>
                    <div style="font-weight: 700; font-size: 1rem; color: var(--text-primary);">${q.ground_truth}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">
                        ID: ${q.query_id}<br>
                        Lat: ${q.query_lat !== null ? q.query_lat.toFixed(3) : 'N/A'}, Lon: ${q.query_lon !== null ? q.query_lon.toFixed(3) : 'N/A'}
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); padding-top: 6px; border-top: 1px dashed var(--border-color);">
                        P@1: ${(q.p1 || 0).toFixed(1)} &bull; P@5: ${(q.p5 || 0).toFixed(2)} &bull; AP: ${(q.ap || 0).toFixed(2)}
                    </div>
                </div>
                <div class="retrieved-grid">
                    ${retrievedHtml}
                </div>
            `;
            container.appendChild(card);
        });

        const loadMoreBtn = document.getElementById('loadMoreBtn');
        if (renderLimit < filteredData.length) {
            loadMoreBtn.style.display = 'inline-block';
            loadMoreBtn.textContent = `Load More Queries (${filteredData.length - renderLimit} remaining)`;
        } else {
            loadMoreBtn.style.display = 'none';
        }
    }

    function loadMore() {
        renderLimit += 30;
        renderQueries();
    }

    window.onload = init;
</script>
</body>
</html>
"""


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

    html_content = (
        HTML_TEMPLATE.replace("{{TITLE}}", html.escape(benchmark_title))
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
