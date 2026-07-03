import requests
import pandas as pd
import argparse
import time

# Pre-configured taxon presets for remote, sparsely populated biomes
BIOME_PRESETS = {
    "desert": 47792,  # Cactaceae (Cacti family - indicators of arid deserts)
    "tundra": 56184,  # Cladonia (Reindeer lichens - indicators of subarctic tundra)
    "wetland": 50220,  # Sphagnum (Peat mosses - indicators of wetlands, bogs, fens)
    "boreal": 54817,  # Pinaceae (Pine family - indicators of coniferous boreal forest)
    "rainforest": 48865  # Arecaceae (Palm family - indicators of tropical/subtropical rainforests)
}


def fetch_observations(bbox=None, limit=1000, taxon_id=47126):
    """
    Fetches research-grade observations from iNaturalist API.
    
    Args:
        bbox (list): [sw_lat, sw_lng, ne_lat, ne_lng] or None for global search.
        limit (int): Maximum number of observations to fetch.
        taxon_id (int): Taxon filter (default: 47126 for Plantae / Plants).
    """
    base_url = "https://api.inaturalist.org/v1/observations"

    params = {
        "quality_grade": "research",
        "identifications": "most_agree",
        "photos": "true",
        "taxon_id": taxon_id,
        "per_page": 200,
        "order": "desc",
        "order_by": "created_at"
    }

    if bbox:
        sw_lat, sw_lng, ne_lat, ne_lng = bbox
        params["swlat"] = sw_lat
        params["swlng"] = sw_lng
        params["nelat"] = ne_lat
        params["nelng"] = ne_lng
        print(f"Querying iNaturalist API for bounding box {bbox} and Taxon ID {taxon_id}...")
    else:
        print(f"Querying iNaturalist API GLOBALLY for Taxon ID {taxon_id}...")

    results = []
    page = 1

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
        time.sleep(1)  # Polite API usage

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch iNaturalist research-grade observations globally or by bounding box.")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                        help="Bounding box SW_Lat SW_Lng NE_Lat NE_Lng (default: None for global search).")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of observations to fetch.")
    parser.add_argument("--taxon_id", type=int, default=None,
                        help="Specific iNaturalist Taxon ID (default: 47126 for all Plants).")
    parser.add_argument("--preset", type=str, default=None, choices=list(BIOME_PRESETS.keys()),
                        help="Use a pre-configured biome taxon preset (desert, tundra, wetland, boreal, rainforest).")
    parser.add_argument("--out", type=str, default=None, help="Output CSV file path.")
    args = parser.parse_args()

    # Determine Taxon ID
    taxon_id = args.taxon_id
    if args.preset:
        taxon_id = BIOME_PRESETS[args.preset]
        print(f"Preset selected: {args.preset} (Taxon ID: {taxon_id})")
    elif taxon_id is None:
        taxon_id = 47126  # Default to Plantae (Plants)

    # Determine Output File Name
    out_file = args.out
    if not out_file:
        suffix = args.preset if args.preset else f"taxon_{taxon_id}"
        out_file = f"inaturalist_{suffix}_global.csv"

    df = fetch_observations(args.bbox, limit=args.limit, taxon_id=taxon_id)

    if not df.empty:
        df.to_csv(out_file, index=False)
        print(f"\nSaved {len(df)} observations to {out_file}")
        print(df.head())
    else:
        print("No observations fetched.")


if __name__ == "__main__":
    main()
