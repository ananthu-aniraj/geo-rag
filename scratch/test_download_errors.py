import requests
import pandas as pd

csv_path = "/user/aaniraj/home/Documents/Projects/data/google_landmarks_v2/google_landmarks_metadata_sampled_h3_res11_max100.csv"
print(f"Reading {csv_path}...")
df = pd.read_csv(csv_path).dropna(subset=['Image_URL']).head(100)

session = requests.Session()
session.headers.update({
    "User-Agent": "Geo-RAG-Landmark-Downloader/1.0 (aaniraj@home; contact: aaniraj@home.com)"
})

out_dir = "/user/aaniraj/home/Documents/Projects/data/google_landmarks_v2/images"
import os
os.makedirs(out_dir, exist_ok=True)

for idx, row in df.iterrows():
    url = row['Image_URL']
    file_name = row['file_name']
    out_path = os.path.join(out_dir, file_name)
    print(f"\n#{idx} URL: {url} -> {out_path}")
    try:
        r = session.get(url, timeout=10)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(r.content)
            print("Write: Success")
        else:
            print(f"Reason: {r.reason}")
    except Exception as e:
        print(f"Exception: {e}")
