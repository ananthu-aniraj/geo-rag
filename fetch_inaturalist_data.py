import requests
import pandas as pd
import argparse
import time

def fetch_observations(bbox, limit=1000, taxon_id=47126):
    """
    Fetches research-grade observations from iNaturalist API.
    
    Args:
        bbox (list): [sw_lat, sw_lng, ne_lat, ne_lng]
        limit (int): Maximum number of observations to fetch.
        taxon_id (int): Taxon filter (default: 47126 for Plantae / Plants).
    """
    base_url = "https://api.inaturalist.org/v1/observations"
    sw_lat, sw_lng, ne_lat, ne_lng = bbox
    
    params = {
        "quality_grade": "research",
        "identifications": "most_agree",
        "photos": "true",
        "taxon_id": taxon_id,
        "swlat": sw_lat,
        "swlng": sw_lng,
        "nelat": ne_lat,
        "nelng": ne_lng,
        "per_page": 200,
        "order": "desc",
        "order_by": "created_at"
    }
    
    results = []
    page = 1
    
    print(f"Querying iNaturalist API for bounding box: {bbox}...")
    while len(results) < limit:
        params["page"] = page
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code != 200:
            print(f"Error querying API: {response.status_code}")
            break
            
        data = response.json()
        observations = data.get("results", [])
        
        if not observations:
            print("No more observations found.")
            break
            
        for obs in observations:
            if len(results) >= limit:
                break
                
            # Extract coordinates
            location = obs.get("location")
            if not location:
                continue
            lat, lng = map(float, location.split(","))
            
            # Extract image URL (medium resolution)
            photos = obs.get("photos", [])
            if not photos:
                continue
            img_url = photos[0].get("url")
            # Convert default thumbnail/square URL to medium resolution URL
            if img_url and "/square." in img_url:
                img_url = img_url.replace("/square.", "/medium.")
            elif img_url and "/thumb." in img_url:
                img_url = img_url.replace("/thumb.", "/medium.")
                
            # Extract species label
            taxon = obs.get("taxon", {})
            species_name = taxon.get("name", "Unknown")
            common_name = taxon.get("preferred_common_name", "Unknown")
            
            results.append({
                "Photo_ID": str(obs["id"]),
                "Platform": "iNaturalist",
                "Latitude": lat,
                "Longitude": lng,
                "Image_URL": img_url,
                "Scientific_Name": species_name,
                "Common_Name": common_name,
                "Date_Observed": obs.get("observed_on_string", "")
            })
            
        print(f"Page {page}: Fetched {len(results)} observations so far...")
        page += 1
        time.sleep(1) # Polite API usage
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Fetch iNaturalist research-grade plant observations.")
    # Default bounding box around Montpellier region (Southern France / Mediterranean area)
    parser.add_argument("--bbox", type=float, nargs=4, default=[43.5, 3.7, 43.7, 4.0],
                        help="Bounding box SW_Lat SW_Lng NE_Lat NE_Lng (default: Montpellier area)")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of observations to fetch.")
    parser.add_argument("--out", type=str, default="inaturalist_test_subset.csv", help="Output CSV file path.")
    args = parser.parse_args()
    
    df = fetch_observations(args.bbox, limit=args.limit)
    
    if not df.empty:
        df.to_csv(args.out, index=False)
        print(f"\nSaved {len(df)} observations to {args.out}")
        print(df.head())
    else:
        print("No observations fetched.")

if __name__ == "__main__":
    main()
