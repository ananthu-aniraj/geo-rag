import pickle
import numpy as np
import argparse
import os


def create_sample_grid(pkl_path, output_html, top_n=5):
    print(f"Loading clustered data from {pkl_path}...")
    if pkl_path.endswith('.pkl'):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # Assume Parquet
        import pandas as pd
        df = pd.read_parquet(pkl_path)
        data = df.to_dict('records')

    if not data or 'cluster_id' not in data[0]:
        print("Error: Data is not clustered. Please run cluster_images_global.py first.")
        return

    # Group data by cluster and prepare a compact JSON-like structure
    print("Aggregating cluster data for the dashboard...")
    cluster_map = {}
    for item in data:
        c_id = int(item['cluster_id'])
        if c_id not in cluster_map:
            cluster_map[c_id] = {
                "id": c_id,
                "label": item.get('cluster_label', 'Unlabeled'),
                "count": 0,
                "images": [],
                "embeddings": []
            }
        cluster_map[c_id]["count"] += 1
        # Store metadata for similarity sorting later
        cluster_map[c_id]["images"].append({
            "url": item['Image_URL'],
            "id": item['Photo_ID'],
            "emb": item['embedding']
        })

    print(f"Processing {len(cluster_map)} clusters...")

    # Final data to embed in HTML
    dashboard_data = []
    sorted_ids = sorted(cluster_map.keys())

    for c_id in sorted_ids:
        c = cluster_map[c_id]
        # Calculate centroid and find top_n representative images
        embs = np.array([img['emb'] for img in c["images"]])
        centroid = np.mean(embs, axis=0)

        # Cosine similarity
        norm_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = np.dot(norm_embs, norm_centroid)

        sorted_indices = np.argsort(sims)[::-1][:top_n]

        # Keep only the representative samples to save space in the HTML
        samples = []
        for idx in sorted_indices:
            img = c["images"][idx]
            samples.append({
                "url": img["url"],
                "id": img["id"],
                "sim": float(sims[idx])
            })

        dashboard_data.append({
            "id": c_id,
            "label": c["label"],
            "count": c["count"],
            "samples": samples
        })

    # Generate the Dynamic Dashboard HTML
    import json
    json_data = json.dumps(dashboard_data)
    
    # Save data to an external JS file to prevent browser freezing on massive inline scripts
    data_js_path = output_html.replace('.html', '_data.js')
    print(f"Writing data payload to {data_js_path}...")
    with open(data_js_path, 'w', encoding='utf-8') as f:
        f.write(f"const CLUSTER_DATA = {json_data};")

    data_js_filename = os.path.basename(data_js_path)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Geo-RAG Cluster Dashboard</title>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/fuse.js/dist/fuse.basic.min.js"></script>
        <script src="{data_js_filename}"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }}
            .header {{ background: #1a73e8; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .controls {{ display: flex; gap: 10px; margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            input, select {{ padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; flex-grow: 1; }}
            .results-container {{ display: flex; flex-direction: column; gap: 20px; }}
            .cluster-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .cluster-header {{ border-bottom: 2px solid #f0f2f5; padding-bottom: 10px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }}
            .cluster-title {{ font-size: 1.2em; font-weight: bold; color: #333; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
            .image-item {{ text-align: center; font-size: 12px; }}
            .image-item img {{ width: 100%; height: 150px; object-fit: cover; border-radius: 4px; border: 1px solid #eee; }}
            .tag {{ background: #e8f0fe; color: #1967d2; padding: 4px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 500; }}
            .stats {{ color: #5f6368; font-size: 0.9em; }}
            .load-more {{ background: #1a73e8; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 20px auto; display: block; font-size: 16px; }}
            .load-more:hover {{ background: #1557b0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Geo-RAG: Global Cluster Dashboard</h1>
            <p>Exploring {len(dashboard_data)} semantic clusters across millions of images.</p>
        </div>

        <div class="controls">
            <input type="text" id="searchInput" placeholder="Search by Cluster ID or Label (e.g., 'Forest' or '142')..." onkeyup="handleSearch()">
            <select id="sortSelect" onchange="resetAndRender()">
                <option value="id">Sort by ID</option>
                <option value="count">Sort by Size (Large First)</option>
            </select>
        </div>

        <div id="results" class="results-container">
            <!-- Clusters will be rendered here via JS -->
        </div>
        
        <button id="loadMoreBtn" class="load-more" onclick="loadMore()" style="display: none;">Load More</button>

        <script>
            const data = typeof CLUSTER_DATA !== 'undefined' ? CLUSTER_DATA : [];
            let filteredData = [...data];
            let renderLimit = 100;

            function renderResults(append = false) {{
                const container = document.getElementById('results');
                const sortVal = document.getElementById('sortSelect').value;
                
                let toRender = [...filteredData];
                if (sortVal === 'count') {{
                    toRender.sort((a, b) => b.count - a.count);
                }} else {{
                    toRender.sort((a, b) => a.id - b.id);
                }}

                const slice = toRender.slice(append ? renderLimit - 100 : 0, renderLimit);
                
                const html = slice.map(c => `
                    <div class="cluster-card">
                        <div class="cluster-header">
                            <div class="cluster-title">Cluster #${{c.id}}: <span style="color: #1a73e8">${{c.label}}</span></div>
                            <div class="stats">
                                <span class="tag">${{c.count.toLocaleString()}} images</span>
                            </div>
                        </div>
                        <div class="image-grid">
                            ${{c.samples.map((s, i) => `
                                <div class="image-item">
                                    <div style="font-weight: bold; color: ${{i===0 ? '#d93025':'#1a73e8'}}; margin-bottom: 4px;">
                                        ${{i===0 ? 'Centroid' : 'Sample ' + i}}
                                    </div>
                                    <a href="${{s.url}}" target="_blank">
                                        <img src="${{s.url}}" loading="lazy">
                                    </a>
                                    <div style="margin-top: 5px; color: #5f6368;">ID: ${{s.id}}<br>Sim: ${{s.sim.toFixed(4)}}</div>
                                </div>
                            `).join('')}}
                        </div>
                    </div>
                `).join('');
                
                if (append) {{
                    container.innerHTML += html;
                }} else {{
                    container.innerHTML = html;
                }}
                
                const btn = document.getElementById('loadMoreBtn');
                if (toRender.length > renderLimit) {{
                    btn.style.display = 'block';
                }} else {{
                    btn.style.display = 'none';
                }}
            }}
            
            function loadMore() {{
                renderLimit += 100;
                renderResults(true);
            }}

            function resetAndRender() {{
                renderLimit = 100;
                renderResults(false);
            }}

            function handleSearch() {{
                const query = document.getElementById('searchInput').value.toLowerCase();
                if (!query) {{
                    filteredData = [...data];
                }} else {{
                    filteredData = data.filter(c => 
                        c.id.toString().includes(query) || 
                        c.label.toLowerCase().includes(query)
                    );
                }}
                resetAndRender();
            }}

            // Initial render
            resetAndRender();
        </script>
    </body>
    </html>
    """

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Scalable Dashboard saved to: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an HTML grid of representative samples for each cluster.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the clustered .pkl file.")
    parser.add_argument("--out", type=str, default="cluster_samples.html", help="Output HTML file name.")
    parser.add_argument("--top_n", type=int, default=6, help="Number of samples to show per cluster.")
    args = parser.parse_args()

    create_sample_grid(args.pkl, args.out, args.top_n)
