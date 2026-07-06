import requests
import pandas as pd
import argparse
import time

# Map preset biomes to a list of plant & animal search terms to ensure a balanced ecological representation
BIOME_PRESETS = {
    "desert": [
        "Cactaceae",  # Cacti (plants)
        "Camelus",  # Camels (mammals)
        "Larrea tridentata",  # Creosote Bush (desert shrubs)
        "Artemisia tridentata"  # Sagebrush (dry scrublands)
    ],
    "tundra": [
        "Cladonia",  # Reindeer Lichens (polar groundcover)
        "Ursus maritimus",  # Polar Bear (arctic ice/tundra predator)
        "Salix arctica",  # Arctic Willow (tundra plant)
        "Rangifer tarandus"  # Caribou / Reindeer (tundra herbivores)
    ],
    "wetland": [
        "Sphagnum",  # Peat Mosses (indicator of bogs & mires)
        "Typha",  # Reeds / Bulrushes (cattails in freshwater marshes)
        "Alcedo atthis",  # Common Kingfisher (wetland birds)
        "Castor",  # Beavers (freshwater ecosystem architects)
        "Caiman"  # Caimans (wetland reptiles)
    ],
    "boreal": [
        "Pinaceae",  # Pine family (conifers)
        "Alces alces",  # Moose (boreal forest large mammal)
        "Lynx lynx",  # Eurasian Lynx (boreal forest predator)
        "Abies"  # Fir trees (boreal evergreen conifers)
    ],
    "rainforest": [
        "Arecaceae",  # Palm family (tropical plants)
        "Orangutan",  # Orangutans (tropical canopy mammals)
        "Panthera onca",  # Jaguar (rainforest predator)
        "Bromeliaceae"  # Bromeliads (tropical rainforest epiphytes)
    ],
    "polar": [
        "Spheniscidae",  # Penguins (Antarctic coastal ecosystems)
        "Ursus maritimus",  # Polar Bear (Arctic sea ice ecosystems)
        "Aptenodytes forsteri"  # Emperor Penguin (Antarctic ice caps)
    ]
}


def resolve_taxon_id(query_str):
    """
    Queries the iNaturalist Taxa Search API to find the closest matching Taxon ID.
    
    Args:
        query_str (str): The common name or scientific name search string.
        
    Returns:
        int: The best matching Taxon ID, or None if not found.
    """
    search_url = "https://api.inaturalist.org/v1/taxa"
    params = {
        "q": query_str,
        "is_active": "true",
        "per_page": 5
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                # 1. Prioritize exact scientific name match (case-insensitive)
                exact_match = None
                for r in results:
                    if r.get("name", "").lower().strip() == query_str.lower().strip():
                        exact_match = r
                        break

                # 2. Prioritize exact common name match (case-insensitive)
                if not exact_match:
                    for r in results:
                        if r.get("preferred_common_name", "").lower().strip() == query_str.lower().strip():
                            exact_match = r
                            break

                # 3. Fallback to the first result if no exact match is found
                best_match = exact_match if exact_match else results[0]

                print(f" -> Successfully resolved search query '{query_str}' to:")
                print(
                    f"    Name: {best_match.get('name')} ({best_match.get('preferred_common_name', 'No common name')})")
                print(f"    Taxon ID: {best_match.get('id')} | Rank: {best_match.get('rank')}")
                return best_match.get("id")
            else:
                print(f" -> No active taxon matches found on iNaturalist for '{query_str}'")
        else:
            print(f" -> Taxa API returned status code {response.status_code}")
    except Exception as e:
        print(f" -> Connection failed during taxon lookup: {e}")

    return None


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
        print(f"   Querying iNaturalist API for bounding box {bbox} and Taxon ID {taxon_id}...")
    else:
        print(f"   Querying iNaturalist API GLOBALLY for Taxon ID {taxon_id}...")

    results = []
    page = 1

    while len(results) < limit:
        params["page"] = page
        response = requests.get(base_url, params=params, timeout=15)

        if response.status_code != 200:
            print(f"   Error querying API: {response.status_code}")
            break

        data = response.json()
        observations = data.get("results", [])

        if not observations:
            print("   No more observations found.")
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

        print(f"   Page {page}: Fetched {len(results)} observations so far...")
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
                        help="Explicit iNaturalist Taxon ID (skips API query lookup if provided).")
    parser.add_argument("--preset", type=str, default=None, choices=list(BIOME_PRESETS.keys()),
                        help="Use a pre-configured biome taxon preset.")
    parser.add_argument("--query", type=str, default=None,
                        help="Search query to dynamically resolve a Taxon ID (e.g. 'sequoia', 'oak', 'sagebrush').")
    parser.add_argument("--out", type=str, default=None, help="Output CSV file path.")
    args = parser.parse_args()

    # Compile queries and explicit taxon IDs to fetch
    queries_to_fetch = []
    explicit_taxon_ids = []

    if args.taxon_id is not None:
        explicit_taxon_ids.append(args.taxon_id)
    elif args.preset:
        queries_to_fetch = BIOME_PRESETS[args.preset]
        print(
            f"Preset selected: {args.preset} ({len(queries_to_fetch)} balanced indicator species: {queries_to_fetch})")
    elif args.query:
        queries_to_fetch = [args.query]
    else:
        explicit_taxon_ids.append(47126)  # Default to Plantae (Plants)

    dfs = []

    # Case A: Fetching explicit Taxon IDs
    if explicit_taxon_ids:
        limit_per_id = max(1, args.limit // len(explicit_taxon_ids))
        for t_id in explicit_taxon_ids:
            df_taxon = fetch_observations(args.bbox, limit=limit_per_id, taxon_id=t_id)
            if not df_taxon.empty:
                dfs.append(df_taxon)

    # Case B: Resolving and fetching query strings
    elif queries_to_fetch:
        limit_per_query = max(1, args.limit // len(queries_to_fetch))
        for q in queries_to_fetch:
            print(f"\nResolving query '{q}'...")
            resolved_id = resolve_taxon_id(q)
            if not resolved_id:
                print(f"Skipping query '{q}' (could not resolve taxon ID)")
                continue

            df_taxon = fetch_observations(args.bbox, limit=limit_per_query, taxon_id=resolved_id)
            if not df_taxon.empty:
                dfs.append(df_taxon)

    # Concatenate all results
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    # Determine Output File Name
    out_file = args.out
    if not out_file:
        suffix = args.preset if args.preset else (
            args.query.replace(" ", "_") if args.query else f"taxon_{args.taxon_id or 'default'}")
        out_file = f"inaturalist_{suffix}_global.csv"

    if not df.empty:
        df.to_csv(out_file, index=False)
        print(f"\nSaved total of {len(df)} observations to {out_file}")
        print(df.groupby(["Scientific_Name", "Common_Name"]).size().reset_index(name="count"))
    else:
        print("No observations fetched.")


if __name__ == "__main__":
    main()
