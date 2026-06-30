import pickle
import numpy as np
import argparse
import os

MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER'


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
                "images": []
            }
        cluster_map[c_id]["count"] += 1

        photo_id = item.get('Photo_ID')
        if photo_id is not None:
            # Handle float or NaN values
            if isinstance(photo_id, float):
                import math as py_math
                if not py_math.isnan(photo_id):
                    photo_id = str(int(photo_id))
                else:
                    photo_id = ""
            else:
                photo_id = str(photo_id).strip()
                if photo_id.endswith('.0'):
                    photo_id = photo_id[:-2]
        else:
            photo_id = ""

        platform = item.get('Platform', '')
        if platform is not None:
            platform = str(platform).strip()
        else:
            platform = ""

        # Store metadata for similarity sorting later
        cluster_map[c_id]["images"].append({
            "url": item['Image_URL'],
            "id": photo_id,
            "emb": item['embedding'],
            "desc": item.get('cluster_description', ''),
            "lat": float(item.get('Latitude', 0.0)),
            "lon": float(item.get('Longitude', 0.0)),
            "h3": item.get('H3_Cell', ''),
            "platform": platform,
            "captured_at": item.get('Captured_At', '')
        })

    print(f"Processing {len(cluster_map)} clusters...")

    # Final data to embed in HTML
    dashboard_data = []
    sorted_ids = sorted(cluster_map.keys())

    import math
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
                "sim": float(sims[idx]),
                "lat": float(img["lat"]),
                "lon": float(img["lon"]),
                "platform": img["platform"],
                "captured_at": img["captured_at"]
            })

        # Grab description from the centroid sample (highest similarity, sorted_indices[0])
        centroid_desc = c["images"][sorted_indices[0]].get("desc", "")

        # Calculate robust geographic center (handling wrap-around for longitude)
        lats = [img['lat'] for img in c["images"]]
        lons = [img['lon'] for img in c["images"]]

        center_lat = sum(lats) / len(lats)
        x = sum(math.cos(math.radians(lon)) for lon in lons) / len(lons)
        y = sum(math.sin(math.radians(lon)) for lon in lons) / len(lons)
        center_lon = math.degrees(math.atan2(y, x))

        # Calculate unique H3 cells count
        h3_cells = set(img['h3'] for img in c["images"] if img['h3'])
        unique_h3_count = len(h3_cells)

        # Compute H3 cell frequency and centroids
        h3_counts = {}
        for img in c["images"]:
            h3_cell = img.get("h3")
            if h3_cell:
                h3_counts[h3_cell] = h3_counts.get(h3_cell, 0) + 1

        h3_centroids = []
        try:
            import h3
            sorted_h3 = sorted(h3_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            for cell, count in sorted_h3:
                try:
                    h3_lat, h3_lon = h3.cell_to_latlon(cell)
                    h3_centroids.append([float(h3_lat), float(h3_lon), int(count), cell])
                except Exception:
                    continue
        except ImportError:
            # Fallback if h3 library is not present: group by rounded lat/lon
            coord_counts = {}
            for img in c["images"]:
                r_lat = round(img['lat'], 2)
                r_lon = round(img['lon'], 2)
                coord_counts[(r_lat, r_lon)] = coord_counts.get((r_lat, r_lon), 0) + 1
            sorted_coords = sorted(coord_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            for (r_lat, r_lon), count in sorted_coords:
                h3_centroids.append([float(r_lat), float(r_lon), int(count), "N/A"])

        dashboard_data.append({
            "id": c_id,
            "label": c["label"],
            "description": centroid_desc,
            "count": c["count"],
            "unique_h3_count": unique_h3_count,
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "h3_centroids": h3_centroids,
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

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Geo-RAG Cluster Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script src="https://cdn.jsdelivr.net/npm/fuse.js/dist/fuse.basic.min.js"></script>
    <script src="{data_js_filename}"></script>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --accent: #f59e0b;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }}

        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}

        .header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            color: white;
            padding: 32px 40px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-md);
        }}

        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 2.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}

        .header p {{
            margin: 0;
            color: #c7d2fe;
            font-size: 1.1rem;
            font-weight: 400;
        }}

        .controls {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
        }}

        input, select {{
            padding: 10px 16px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 0.95rem;
            font-family: inherit;
            color: var(--text-main);
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}

        input {{
            flex-grow: 1;
        }}

        input:focus, select:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }}

        .results-container {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}

        .cluster-card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .cluster-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }}

        .cluster-header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .cluster-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #1e293b;
        }}

        .cluster-title span {{
            color: var(--primary);
        }}

        .stats-badges {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .tag {{
            background: #e0e7ff;
            color: #4338ca;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .tag-geo {{
            background: #fef3c7;
            color: #d97706;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .desc-box {{
            margin-bottom: 20px;
            color: var(--text-muted);
            font-size: 0.95rem;
            padding: 14px 18px;
            background: #f8fafc;
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            line-height: 1.5;
        }}

        .desc-box b {{
            color: var(--text-main);
        }}

        .map-toggle-btn {{
            background-color: #f1f5f9;
            color: #334155;
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            margin-bottom: 16px;
        }}

        .map-toggle-btn:hover {{
            background-color: #e2e8f0;
            color: #0f172a;
        }}

        .map-toggle-btn.active {{
            background-color: var(--primary);
            color: white;
            border-color: var(--primary);
        }}

        .map-container {{
            height: 350px;
            width: 100%;
            margin-bottom: 20px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 20px;
        }}

        .image-item {{
            display: flex;
            flex-direction: column;
            background: #f8fafc;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            transition: box-shadow 0.2s;
        }}

        .image-item:hover {{
            box-shadow: var(--shadow-sm);
        }}

        .image-role {{
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }}

        .image-item img {{
            width: 100%;
            height: 130px;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin-bottom: 8px;
        }}

        .image-meta {{
            font-size: 0.78rem;
            color: var(--text-muted);
            line-height: 1.4;
            margin-top: auto;
        }}

        .load-more {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            cursor: pointer;
            margin: 32px auto;
            display: block;
            font-size: 1rem;
            font-weight: 600;
            box-shadow: var(--shadow-sm);
            transition: background-color 0.2s;
        }}

        .load-more:hover {{
            background-color: var(--primary-hover);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Geo-RAG: Global Cluster Dashboard</h1>
        <p>Exploring {len(dashboard_data)} semantic clusters with geographic distributions across global streetscapes.</p>
    </div>

    <div class="controls">
        <input type="text" id="searchInput" placeholder="Search by Cluster ID, label description, or coordinates..." onkeyup="handleSearch()">
        <select id="sortSelect" onchange="resetAndRender()">
            <option value="id">Sort by ID</option>
            <option value="count">Sort by Size (Large First)</option>
            <option value="geo">Sort by Geographic Spread</option>
        </select>
    </div>

    <div id="results" class="results-container">
        <!-- Clusters rendered here via JS -->
    </div>
    
    <button id="loadMoreBtn" class="load-more" onclick="loadMore()" style="display: none;">Load More Clusters</button>

    <script>
        const MAPILLARY_TOKEN = 'MAPILLARY_TOKEN_PLACEHOLDER';
        const data = typeof CLUSTER_DATA !== 'undefined' ? CLUSTER_DATA : [];
        let filteredData = [...data];
        let renderLimit = 50;
        const activeMaps = {{}};

        function handleImageError(img, photoId, platform) {{
            // Avoid infinite loops if retry fails
            if (img.dataset.retryAttempt) return;
            img.dataset.retryAttempt = '1';
            
            const platformLower = String(platform).toLowerCase().trim();
            const isMapillary = platformLower === 'mapillary' || img.src.includes('mapillary') || img.src.includes('fbcdn.net');
            const isKartaview = platformLower === 'kartaview' || img.src.includes('kartaview') || img.src.includes('openstreetcam');
            
            // Clean up photoId to remove any trailing .0 or other formatting issues
            let cleanPhotoId = String(photoId).trim();
            if (cleanPhotoId.endsWith('.0')) {{
                cleanPhotoId = cleanPhotoId.slice(0, -2);
            }}
            
            if (isMapillary && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {{
                const apiUrl = `https://graph.mapillary.com/${{cleanPhotoId}}?fields=thumb_1024_url`;
                fetch(apiUrl, {{
                    headers: {{ 'Authorization': `OAuth ${MAPILLARY_TOKEN}` }}
                }})
                .then(res => res.json())
                .then(resData => {{
                    if (resData.thumb_1024_url) {{
                        img.src = resData.thumb_1024_url;
                        // Also update containing link if applicable
                        const link = img.closest('a');
                        if (link) link.href = resData.thumb_1024_url;
                    }}
                }})
                .catch(err => console.error('Error fetching Mapillary fresh URL:', err));
            }} else if (isKartaview && cleanPhotoId && cleanPhotoId !== 'null' && cleanPhotoId !== 'undefined' && cleanPhotoId !== 'NaN') {{
                const apiUrl = `https://api.openstreetcam.org/2.0/photo/${{cleanPhotoId}}`;
                fetch(apiUrl)
                .then(res => res.json())
                .then(resData => {{
                    const data = resData.result && resData.result.data;
                    if (data) {{
                        const freshUrl = data.fileurlLTh || data.fileurlTh || data.fileurl;
                        if (freshUrl) {{
                            img.src = freshUrl;
                            const link = img.closest('a');
                            if (link) link.href = freshUrl;
                        }}
                    }}
                }})
                .catch(err => console.error('Error fetching Kartaview fresh URL:', err));
            }}
        }}

        function renderResults(append = false) {{
            const container = document.getElementById('results');
            const sortVal = document.getElementById('sortSelect').value;
            
            let toRender = [...filteredData];
            if (sortVal === 'count') {{
                toRender.sort((a, b) => b.count - a.count);
            }} else if (sortVal === 'geo') {{
                toRender.sort((a, b) => b.unique_h3_count - a.unique_h3_count);
            }} else {{
                toRender.sort((a, b) => a.id - b.id);
            }}

            const slice = toRender.slice(append ? renderLimit - 50 : 0, renderLimit);
            
            const html = slice.map(c => `
                <div class="cluster-card" id="cluster-card-${{c.id}}">
                    <div class="cluster-header">
                        <div class="cluster-title">Cluster #${{c.id}}: <span>${{c.label}}</span></div>
                        <div class="stats-badges">
                            <span class="tag">${{c.count.toLocaleString()}} images</span>
                            <span class="tag-geo">${{c.unique_h3_count.toLocaleString()}} H3 cells</span>
                        </div>
                    </div>
                    ${{c.description ? `
                    <div class="desc-box">
                        <b>VLM Prototypical Description:</b> ${{c.description}}
                    </div>` : ''}}
                    
                    <button class="map-toggle-btn" id="map-btn-${{c.id}}" onclick="toggleMap(${{c.id}})">
                        🗺️ Show Geographic Spread Map
                    </button>
                    
                    <div id="map-container-${{c.id}}" class="map-container" style="display: none;"></div>

                    <div class="image-grid">
                        ${{c.samples.map((s, i) => `
                            <div class="image-item">
                                <div class="image-role" style="color: ${{i===0 ? '#d93025':'#4f46e5'}}">
                                    ${{i===0 ? 'Centroid Image' : 'Representative Sample ' + i}}
                                </div>
                                <a href="${{s.url}}" target="_blank">
                                    <img src="${{s.url}}" onerror="handleImageError(this, '${{s.id}}', '${{s.platform}}')" loading="lazy">
                                </a>
                                <div class="image-meta">
                                    <b>ID:</b> ${{s.id}}<br>
                                    <b>Similarity:</b> ${{s.sim.toFixed(4)}}<br>
                                    <b>Lat/Lon:</b> ${{s.lat.toFixed(4)}}, ${{s.lon.toFixed(4)}}${{s.captured_at ? '<br><b>Taken:</b> ' + s.captured_at : ''}}
                                </div>
                            </div>
                        `).join('')}}
                    </div>
                </div>
            `).join('');
            
            if (append) {{
                container.innerHTML += html;
            }} else {{
                // Cleanup maps that are no longer visible
                Object.keys(activeMaps).forEach(id => {{
                    activeMaps[id].remove();
                    delete activeMaps[id];
                }});
                container.innerHTML = html;
            }}
            
            const btn = document.getElementById('loadMoreBtn');
            if (toRender.length > renderLimit) {{
                btn.style.display = 'block';
            }} else {{
                btn.style.display = 'none';
            }}
        }}

        function toggleMap(clusterId) {{
            const container = document.getElementById(`map-container-${{clusterId}}`);
            const btn = document.getElementById(`map-btn-${{clusterId}}`);
            
            if (container.style.display === 'none') {{
                container.style.display = 'block';
                btn.innerHTML = '🗺️ Hide Geographic Map';
                btn.classList.add('active');
                initMap(clusterId);
            }} else {{
                container.style.display = 'none';
                btn.innerHTML = '🗺️ Show Geographic Spread Map';
                btn.classList.remove('active');
            }}
        }}

        function initMap(clusterId) {{
            if (activeMaps[clusterId]) {{
                activeMaps[clusterId].invalidateSize();
                return;
            }}

            const cluster = data.find(c => c.id === clusterId);
            if (!cluster) return;

            // Instantiate map
            const map = L.map(`map-container-${{clusterId}}`).setView([cluster.center_lat, cluster.center_lon], 2);
            activeMaps[clusterId] = map;

            // Load CartoDB Positron tiles
            L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
            }}).addTo(map);

            // Add circle for cluster geographical center
            L.circle([cluster.center_lat, cluster.center_lon], {{
                color: '#4f46e5',
                fillColor: '#4f46e5',
                fillOpacity: 0.1,
                radius: 100000 // 100km radius
            }}).addTo(map).bindPopup(`<b>Cluster center</b><br>Lat: ${{cluster.center_lat.toFixed(4)}}, Lon: ${{cluster.center_lon.toFixed(4)}}`);

            // Add density circle markers for H3 centroids
            let maxCount = 1;
            cluster.h3_centroids.forEach(pt => {{
                if (pt[2] > maxCount) maxCount = pt[2];
            }});

            cluster.h3_centroids.forEach(pt => {{
                const lat = pt[0];
                const lon = pt[1];
                const count = pt[2];
                const cell = pt[3];
                const percentage = (count / cluster.count) * 100;

                L.circleMarker([lat, lon], {{
                    radius: Math.max(5, Math.min(25, 5 + (count / maxCount) * 15)),
                    fillColor: '#f59e0b',
                    color: '#d97706',
                    weight: 1,
                    opacity: 0.8,
                    fillOpacity: 0.5
                }}).addTo(map).bindPopup(`
                    <b>Location Cluster (H3: ${{cell}})</b><br>
                    Images: ${{count.toLocaleString()}} (${{percentage.toFixed(2)}}% of cluster)
                `);
            }});

            // Add custom markers for the representative sample images
            cluster.samples.forEach((s, idx) => {{
                const markerColor = idx === 0 ? '#d93025' : '#4f46e5';
                const label = idx === 0 ? 'Centroid image location' : `Sample ${{idx}} location`;
                
                L.circleMarker([s.lat, s.lon], {{
                    radius: 8,
                    fillColor: markerColor,
                    color: '#ffffff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.9
                }}).addTo(map).bindPopup(`
                    <div style="width: 140px; text-align: center;">
                        <b>${{label}}</b><br>
                        <img src="${{s.url}}" onerror="handleImageError(this, '${{s.id}}', '${{s.platform}}')" style="width: 100%; height: 90px; object-fit: cover; margin-top: 6px; border-radius: 4px; border: 1px solid #ddd;"><br>
                        ${{s.captured_at ? '<span style="font-size: 10px; color: #555;"><b>Taken:</b> ' + s.captured_at + '</span><br>' : ''}}
                        <a href="${{s.url}}" target="_blank" style="font-size: 11px; color: #4f46e5; font-weight: 600; text-decoration: none;">View Original</a>
                    </div>
                `);
            }});

            // Adjust viewport to show all geographic locations
            const group = new L.featureGroup();
            cluster.samples.forEach(s => {{
                group.addLayer(L.marker([s.lat, s.lon]));
            }});
            cluster.h3_centroids.forEach(pt => {{
                group.addLayer(L.marker([pt[0], pt[1]]));
            }});
            if (group.getLayers().length > 0) {{
                map.fitBounds(group.getBounds().pad(0.15));
            }}
        }}
        
        function loadMore() {{
            renderLimit += 50;
            renderResults(true);
        }}

        function resetAndRender() {{
            renderLimit = 50;
            renderResults(false);
        }}

        function handleSearch() {{
            const query = document.getElementById('searchInput').value.toLowerCase();
            if (!query) {{
                filteredData = [...data];
            }} else {{
                filteredData = data.filter(c => 
                    c.id.toString().includes(query) || 
                    c.label.toLowerCase().includes(query) ||
                    (c.description && c.description.toLowerCase().includes(query)) ||
                    c.center_lat.toFixed(2).includes(query) ||
                    c.center_lon.toFixed(2).includes(query)
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
