import pickle
import numpy as np
import argparse
import os

def create_sample_grid(pkl_path, output_html, top_n=5):
    print(f"Loading clustered data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if not data or 'cluster_id' not in data[0]:
        print("Error: Data is not clustered. Please run cluster_images_global.py first.")
        return

    # Group data by cluster
    clusters = {}
    for item in data:
        c_id = item['cluster_id']
        if c_id not in clusters:
            clusters[c_id] = []
        clusters[c_id].append(item)

    # Sort cluster IDs
    sorted_cluster_ids = sorted(clusters.keys())

    html_content = f"""
    <html>
    <head>
        <title>Cluster Representative Samples</title>
        <style>
            body {{ font-family: sans-serif; background-color: #f4f4f4; padding: 20px; }}
            .cluster-container {{ background-color: white; margin-bottom: 30px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .cluster-title {{ border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; color: #333; }}
            .image-grid {{ display: flex; flex-wrap: wrap; gap: 15px; }}
            .image-card {{ width: 200px; text-align: center; font-size: 12px; }}
            .image-card img {{ width: 200px; height: 150px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }}
            .centroid-label {{ color: #d9534f; font-weight: bold; margin-bottom: 5px; }}
            .sample-label {{ color: #5bc0de; font-weight: bold; margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <h1>Geo-RAG: Global Cluster Representative Samples</h1>
        <p>Showing the image closest to the centroid and {top_n-1} other representative samples per cluster.</p>
    """

    for c_id in sorted_cluster_ids:
        items = clusters[c_id]
        # Get the label from the first item (all items in cluster share the same label)
        c_name = items[0].get('cluster_label', 'Unlabeled')
        
        # Calculate centroid
        embs = np.array([item['embedding'] for item in items]).squeeze()
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        centroid = np.mean(embs, axis=0)
        
        # Calculate distances to centroid (Cosine similarity dot product on normalized vectors)
        norm_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        similarities = np.dot(norm_embs, norm_centroid)
        
        # Sort items by similarity to centroid
        sorted_indices = np.argsort(similarities)[::-1]
        
        html_content += f"""
        <div class="cluster-container">
            <h2 class="cluster-title">Cluster {c_id}: {c_name} ({len(items)} images)</h2>
            <div class="image-grid">
        """
        
        for i in range(min(top_n, len(items))):
            idx = sorted_indices[i]
            item = items[idx]
            label = "Centroid (Closest)" if i == 0 else f"Sample {i}"
            label_class = "centroid-label" if i == 0 else "sample-label"
            
            html_content += f"""
                <div class="image-card">
                    <div class="{label_class}">{label}</div>
                    <a href="{item['Image_URL']}" target="_blank">
                        <img src="{item['Image_URL']}" alt="Cluster {c_id} Sample">
                    </a>
                    <p>ID: {item['Photo_ID']}<br>Sim: {similarities[idx]:.4f}</p>
                </div>
            """
            
        html_content += """
            </div>
        </div>
        """

    html_content += "</body></html>"

    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"Sample grid saved to: {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an HTML grid of representative samples for each cluster.")
    parser.add_argument("--pkl", type=str, required=True, help="Path to the clustered .pkl file.")
    parser.add_argument("--out", type=str, default="cluster_samples.html", help="Output HTML file name.")
    parser.add_argument("--top_n", type=int, default=6, help="Number of samples to show per cluster.")
    args = parser.parse_args()

    create_sample_grid(args.pkl, args.out, args.top_n)
