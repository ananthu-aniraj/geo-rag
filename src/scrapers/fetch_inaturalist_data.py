import argparse
import re
import time

import pandas as pd
import requests

# Map preset biomes to a list of plant & animal search terms to ensure a balanced ecological representation
BIOME_PRESETS = {
    "desert": [
        "Cactaceae",  # Cacti (plants)
        "Camelus",  # Camels (mammals)
        "Larrea tridentata",  # Creosote Bush (desert shrubs)
        "Artemisia tridentata",  # Sagebrush (dry scrublands)
        "Struthio camelus",  # Ostrich (desert birds)
        "Struthio molybdophanes",  # Somali Ostrich (desert birds)
    ],
    "tundra": [
        "Cladonia",  # Reindeer Lichens (polar groundcover)
        "Ursus maritimus",  # Polar Bear (arctic ice/tundra predator)
        "Salix arctica",  # Arctic Willow (tundra plant)
        "Rangifer tarandus",  # Caribou / Reindeer (tundra herbivores)
    ],
    "wetland": [
        "Sphagnum",  # Peat Mosses (indicator of bogs & mires)
        "Typha",  # Reeds / Bulrushes (cattails in freshwater marshes)
        "Alcedo atthis",  # Common Kingfisher (wetland birds)
        "Castor",  # Beavers (freshwater ecosystem architects)
        "Caiman",  # Caimans (wetland reptiles)
    ],
    "boreal": [
        "Pinaceae",  # Pine family (conifers)
        "Alces alces",  # Moose (boreal forest large mammal)
        "Lynx lynx",  # Eurasian Lynx (boreal forest predator)
        "Abies",  # Fir trees (boreal evergreen conifers)
    ],
    "rainforest": [
        "Arecaceae",  # Palm family (tropical plants)
        "Orangutan",  # Orangutans (tropical canopy mammals)
        "Panthera onca",  # Jaguar (rainforest predator)
        "Bromeliaceae",  # Bromeliads (tropical rainforest epiphytes)
    ],
    "polar": [
        "Spheniscidae",  # Penguins (Antarctic coastal ecosystems)
        "Ursus maritimus",  # Polar Bear (Arctic sea ice ecosystems)
        "Aptenodytes forsteri",  # Emperor Penguin (Antarctic ice caps)
    ],
}


def scrape_species_from_wikipedia(biome_name, limit=10):
    """
    Searches Wikipedia for species lists associated with a biome or ecosystem
    and extracts binomial scientific names.
    """
    search_url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Try different search terms in a fallback loop to maximize probability of page discovery
    search_queries = [
        f"List of {biome_name} plants",
        f"List of {biome_name} species",
        f"{biome_name} flora",
        f"{biome_name} plants",
        f"{biome_name}",
    ]

    search_results = []
    for q in search_queries:
        print(f" -> Searching Wikipedia for: '{q}'...")
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": q,
            "format": "json",
        }
        try:
            res = requests.get(
                search_url, params=search_params, headers=headers, timeout=10
            )
            if res.status_code == 200:
                results = res.json().get("query", {}).get("search", [])
                if results:
                    search_results = results
                    break
            else:
                print(f"    (Wikipedia returned HTTP status code {res.status_code})")
        except Exception as e:
            print(f"    (Wikipedia request failed: {e})")

    if not search_results:
        print(" -> No Wikipedia page matches found for any search queries.")
        return []

    page_title = search_results[0]["title"]
    print(f" -> Found Wikipedia page: '{page_title}'")

    # Fetch full page wikitext content (revisions)
    content_params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "titles": page_title,
        "format": "json",
    }

    try:
        res_content = requests.get(
            search_url, params=content_params, headers=headers, timeout=10
        )
        if res_content.status_code != 200:
            print(f" -> Failed to fetch content: HTTP {res_content.status_code}")
            return []

        pages = res_content.json().get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0]
        revisions = pages[page_id].get("revisions", [])
        if not revisions:
            return []

        # Extract full wikitext (standard content is under '*')
        text = revisions[0].get("*", "")
        if not text:
            # Fallback if modern slots structure is returned
            slots = revisions[0].get("slots", {})
            text = slots.get("main", {}).get("*", "")

        if not text:
            print(" -> Page content is empty.")
            return []

        # Extract binomial scientific names (Genus species)
        # Capitalized genus name + lowercase species name of length 3-20
        binomial_pattern = re.compile(r"\b([A-Z][a-z]+ [a-z]{3,20})\b")
        potential_names = binomial_pattern.findall(text)

        # Deduplicate
        unique_names = list(set(potential_names))

        # Exclude common false positive English words that look like binomial names
        exclude_set = {
            "North",
            "South",
            "East",
            "West",
            "New",
            "The",
            "In",
            "On",
            "Of",
            "For",
            "With",
            "United",
            "States",
            "America",
            "Africa",
            "Europe",
            "Asia",
            "Australia",
            "Pacific",
            "Atlantic",
            "Indian",
            "Ocean",
            "Sea",
            "National",
            "Park",
            "Forest",
            "Desert",
            "Common",
            "List",
            "Flora",
            "Fauna",
            "Species",
            "Genus",
            "Family",
            "Order",
            "Class",
            "This",
            "That",
            "These",
            "Those",
            "Under",
            "Between",
            "During",
            "Throughout",
            "Although",
            "Main",
            "Article",
            "Many",
            "Some",
            "Most",
            "Also",
            "From",
            "Here",
            "There",
        }

        cleaned_names = [
            n for n in unique_names if not any(w in n for w in exclude_set)
        ]
        print(
            f" -> Extracted {len(cleaned_names)} potential scientific names from Wikipedia."
        )
        return cleaned_names
    except Exception as e:
        print(f" -> Wikipedia content extraction failed: {e}")

    return []


def resolve_place_id(place_name):
    """
    Queries the iNaturalist Places autocomplete API to resolve a country or region to its Place ID.
    """
    url = "https://api.inaturalist.org/v1/places/autocomplete"
    params = {"q": place_name}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                # Prioritize exact or country-level matches
                best_match = results[0]
                print(f" -> Resolved place '{place_name}' to:")
                print(f"    Display Name: {best_match.get('display_name')}")
                print(
                    f"    Place ID: {best_match.get('id')} | Slug: {best_match.get('slug')}"
                )
                return best_match.get("id")
            else:
                print(f" -> No place matches found on iNaturalist for '{place_name}'")
        else:
            print(f" -> Places API returned status code {response.status_code}")
    except Exception as e:
        print(f" -> Connection failed during place lookup: {e}")

    return None


def fetch_top_species_for_place(
    place_id, taxon_id=47126, without_taxon_id=None, limit=10
):
    """
    Queries iNaturalist Species Counts API to get the most commonly observed species in a place.
    """
    url = "https://api.inaturalist.org/v1/observations/species_counts"
    params = {"place_id": place_id, "taxon_id": taxon_id, "per_page": limit}
    if without_taxon_id:
        params["without_taxon_id"] = without_taxon_id

    species_list = []
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f" -> Discovered top {len(results)} species for Place ID {place_id}:")
            for item in results:
                taxon = item.get("taxon", {})
                species_id = taxon.get("id")
                name = taxon.get("name")
                common = taxon.get("preferred_common_name", "No common name")
                count = item.get("count", 0)
                if species_id:
                    print(f"    - {name} ({common}) | {count} observations")
                    species_list.append(
                        {"id": species_id, "name": name, "common": common}
                    )
        else:
            print(f" -> Species counts API returned status code {response.status_code}")
    except Exception as e:
        print(f" -> Connection failed during species counts fetch: {e}")

    return species_list


def resolve_taxon_id(query_str):
    """
    Queries the iNaturalist Taxa Search API to find the closest matching Taxon ID.
    """
    search_url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": query_str, "is_active": "true", "per_page": 5}

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
                        if (
                            r.get("preferred_common_name", "").lower().strip()
                            == query_str.lower().strip()
                        ):
                            exact_match = r
                            break

                # 3. Fallback to the first result if no exact match is found
                best_match = exact_match if exact_match else results[0]

                print(f" -> Successfully resolved search query '{query_str}' to:")
                print(
                    f"    Name: {best_match.get('name')} ({best_match.get('preferred_common_name', 'No common name')})"
                )
                print(
                    f"    Taxon ID: {best_match.get('id')} | Rank: {best_match.get('rank')}"
                )
                return best_match.get("id")
            else:
                # Silently return None so loop validation works cleanly
                pass
        else:
            pass
    except Exception:
        pass

    return None


def fetch_observations(
    bbox=None, place_id=None, limit=1000, taxon_id=47126, without_taxon_id=None
):
    """
    Fetches research-grade observations from iNaturalist API.
    """
    base_url = "https://api.inaturalist.org/v1/observations"

    params = {
        "quality_grade": "research",
        "identifications": "most_agree",
        "photos": "true",
        "taxon_id": taxon_id,
        "per_page": 200,
        "order": "desc",
        "order_by": "created_at",
    }
    if without_taxon_id:
        params["without_taxon_id"] = without_taxon_id

    if bbox:
        sw_lat, sw_lng, ne_lat, ne_lng = bbox
        params["swlat"] = sw_lat
        params["swlng"] = sw_lng
        params["nelat"] = ne_lat
        params["nelng"] = ne_lng
        print(
            f"   Querying iNaturalist API for bounding box {bbox} and Taxon ID {taxon_id}..."
        )
    elif place_id:
        params["place_id"] = place_id
        print(
            f"   Querying iNaturalist API for Place ID {place_id} and Taxon ID {taxon_id}..."
        )
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

            results.append(
                {
                    "Photo_ID": str(obs["id"]),
                    "Platform": "iNaturalist",
                    "Latitude": lat,
                    "Longitude": lng,
                    "Image_URL": img_url,
                    "Scientific_Name": species_name,
                    "Common_Name": common_name,
                    "Date_Observed": obs.get("observed_on_string", ""),
                }
            )

        print(f"   Page {page}: Fetched {len(results)} observations so far...")
        page += 1
        time.sleep(1)  # Polite API usage

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch iNaturalist research-grade observations globally or by bounding box."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        help="Bounding box SW_Lat SW_Lng NE_Lat NE_Lng (default: None for global search).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of observations to fetch.",
    )
    parser.add_argument(
        "--taxon_id",
        type=int,
        default=None,
        help="Explicit iNaturalist Taxon ID (skips API query lookup if provided).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(BIOME_PRESETS.keys()),
        help="Use a pre-configured biome taxon preset.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query to dynamically resolve a Taxon ID (e.g. 'sequoia', 'oak', 'sagebrush').",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Name of a country or region to target (resolves Place ID and gathers most observed native species dynamically).",
    )
    parser.add_argument(
        "--num_species",
        type=int,
        default=10,
        help="Number of top native species to dynamically discover and balance when --country is specified (default: 10).",
    )
    parser.add_argument(
        "--target_taxon",
        type=str,
        default="plants",
        help="The taxon kingdom/group for dynamic species counts discovery (e.g. 'plants', 'animals', 'birds', 'insects').",
    )
    parser.add_argument(
        "--exclude_flying",
        action="store_true",
        help="Exclude flying animals (specifically Birds and Insects) from dynamic discovery and queries.",
    )
    parser.add_argument(
        "--scrape_wiki",
        action="store_true",
        help="Treat --query or --preset as a biome, scrape Wikipedia for species names, and fetch them.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output CSV file path.")
    args = parser.parse_args()

    # 1. Resolve Country/Place boundary if specified
    place_id = None
    if args.country:
        print(f"Resolving Place ID for country/region '{args.country}'...")
        place_id = resolve_place_id(args.country)
        if not place_id:
            print(
                "Failed to resolve country. Running without geographic place boundaries."
            )

    # 2. Determine target taxon exclusions
    without_taxon_id = None
    if args.exclude_flying:
        # Exclude Aves (3) and Insecta (47158)
        without_taxon_id = "3,47158"
        print(
            "Flag active: Excluding flying animal kingdoms (Birds and Insects) from observations."
        )

    dfs = []

    # 3. Dynamic Wikipedia Biome Scraping
    if args.scrape_wiki and (args.query or args.preset):
        biome_name = args.preset if args.preset else args.query
        scraped_names = scrape_species_from_wikipedia(
            biome_name, limit=args.num_species
        )

        validated_taxon_ids = []
        validated_names = []

        if scraped_names:
            print("\nValidating Wikipedia species names on iNaturalist API...")
            for name in scraped_names:
                if len(validated_taxon_ids) >= args.num_species:
                    break
                resolved_id = resolve_taxon_id(name)
                if resolved_id:
                    validated_taxon_ids.append(resolved_id)
                    validated_names.append(name)
                    time.sleep(0.1)

            print(
                f"Validated {len(validated_taxon_ids)} species out of {len(scraped_names)} candidates."
            )
        else:
            print("No species names could be extracted from Wikipedia.")

        if validated_taxon_ids:
            limit_per_species = max(1, args.limit // len(validated_taxon_ids))
            for i, t_id in enumerate(validated_taxon_ids):
                name = validated_names[i]
                print(
                    f"\nFetching observations for dynamically resolved Wikipedia species '{name}'..."
                )
                df_sp = fetch_observations(
                    bbox=args.bbox,
                    place_id=place_id,
                    limit=limit_per_species,
                    taxon_id=t_id,
                    without_taxon_id=without_taxon_id,
                )
                if not df_sp.empty:
                    dfs.append(df_sp)
        else:
            print(
                "No Wikipedia species could be validated. Falling back to default search..."
            )

    # 4. Dynamic Country Species Discovery
    elif place_id and not args.taxon_id and not args.preset and not args.query:
        print(f"Resolving target taxon '{args.target_taxon}'...")
        target_taxon_id = resolve_taxon_id(args.target_taxon)
        if not target_taxon_id:
            print(
                f"Could not resolve target taxon '{args.target_taxon}'. Falling back to Plantae (Plants) (ID: 47126)."
            )
            target_taxon_id = 47126

        print(
            f"\nDynamically discovering the top {args.num_species} species under target taxon '{args.target_taxon}' (ID: {target_taxon_id}) in {args.country}..."
        )
        top_species = fetch_top_species_for_place(
            place_id,
            taxon_id=target_taxon_id,
            without_taxon_id=without_taxon_id,
            limit=args.num_species,
        )

        if top_species:
            limit_per_species = max(1, args.limit // len(top_species))
            for sp in top_species:
                print(
                    f"\nFetching observations for '{sp['name']}' ({sp['common']}) in {args.country}..."
                )
                df_sp = fetch_observations(
                    bbox=None,
                    place_id=place_id,
                    limit=limit_per_species,
                    taxon_id=sp["id"],
                    without_taxon_id=without_taxon_id,
                )
                if not df_sp.empty:
                    dfs.append(df_sp)
        else:
            print(
                f"Failed to discover species list. Falling back to downloading generic taxon (ID: {target_taxon_id}) for the place..."
            )
            df_generic = fetch_observations(
                bbox=None,
                place_id=place_id,
                limit=args.limit,
                taxon_id=target_taxon_id,
                without_taxon_id=without_taxon_id,
            )
            if not df_generic.empty:
                dfs.append(df_generic)

    # 5. Standard queries (With/Without Country Place Filter)
    else:
        queries_to_fetch = []
        explicit_taxon_ids = []

        if args.taxon_id is not None:
            explicit_taxon_ids.append(args.taxon_id)
        elif args.preset:
            queries_to_fetch = BIOME_PRESETS[args.preset]
            print(
                f"Preset selected: {args.preset} ({len(queries_to_fetch)} balanced indicator species: {queries_to_fetch})"
            )
        elif args.query:
            queries_to_fetch = [args.query]
        else:
            explicit_taxon_ids.append(47126)  # Default to Plantae (Plants)

        if explicit_taxon_ids:
            limit_per_id = max(1, args.limit // len(explicit_taxon_ids))
            for t_id in explicit_taxon_ids:
                df_taxon = fetch_observations(
                    bbox=args.bbox,
                    place_id=place_id,
                    limit=limit_per_id,
                    taxon_id=t_id,
                    without_taxon_id=without_taxon_id,
                )
                if not df_taxon.empty:
                    dfs.append(df_taxon)

        elif queries_to_fetch:
            limit_per_query = max(1, args.limit // len(queries_to_fetch))
            for q in queries_to_fetch:
                print(f"\nResolving query '{q}'...")
                resolved_id = resolve_taxon_id(q)
                if not resolved_id:
                    print(f"Skipping query '{q}' (could not resolve taxon ID)")
                    continue

                df_taxon = fetch_observations(
                    bbox=args.bbox,
                    place_id=place_id,
                    limit=limit_per_query,
                    taxon_id=resolved_id,
                    without_taxon_id=without_taxon_id,
                )
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
        suffix = ""
        if args.country:
            suffix += args.country.replace(" ", "_")
        if args.preset:
            suffix += f"_{args.preset}" if suffix else args.preset
        elif args.query:
            q_suf = args.query.replace(" ", "_")
            suffix += f"_{q_suf}" if suffix else q_suf
        elif args.taxon_id:
            suffix += f"_taxon_{args.taxon_id}" if suffix else f"taxon_{args.taxon_id}"

        if args.scrape_wiki:
            suffix += "_wiki_scraped"

        if not suffix:
            suffix = "taxon_default"

        out_file = f"inaturalist_{suffix}_global.csv"

    if not df.empty:
        df.to_csv(out_file, index=False)
        print(f"\nSaved total of {len(df)} observations to {out_file}")
        print(
            df.groupby(["Scientific_Name", "Common_Name"])
            .size()
            .reset_index(name="count")
        )
    else:
        print("No observations fetched.")


if __name__ == "__main__":
    main()
