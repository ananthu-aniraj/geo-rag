# Step 2b & 2c: Multi-Modal LLM Auto-Labeling

This document describes the design and prompting strategies of `label_clusters_mllm.py` and `relabel_failed_clusters.py`, which handle automated semantic labeling and fallback depth retries.

---

## ⚙️ SGLang Docker Lifecycle Management

To run local vision-language model inference on GPUs, the master script `run_full_pipeline.sh` automatically manages the lifecycle of the SGLang container (`sglang-server`):
1. **Pre-Launch Cleanup**: Scan and kill any stale container bindings to clear Nvidia GPU memory and ports.
2. **Container Launch**: Spawns the container with NVIDIA runtime support:
   ```bash
   docker run -d --gpus all -p 30000:30000 --name sglang-server ...
   ```
3. **Health Checks**: Polls the server `/health` endpoint and monitors container status until the backend is fully initialized.
4. **Autonomous Teardown**: Automatically kills and removes the container upon completion or pipeline interruption (via bash `trap` handlers).

---

## 🤖 Two-Step Prompting Strategy
### 📷 Step 1: Multimodal Visual Description
* **Input**: The representative cluster image + `prompts/shared/prompt_step1.txt`
* **Output**: A detailed, objective paragraph describing the physical scene features (no LULC labels).

**Prompt Template (`prompts/shared/prompt_step1.txt`):**
```text
You are analyzing a single ground-level photograph from a global land-survey campaign; the sample point may be anywhere on Earth (tropical, arid, temperate, boreal, alpine, wetland, or coastal). The camera stands at a sample point and looks outward; the land cover AT THE FOREGROUND/CENTER of the frame is the DOMINANT cover that must be characterized. Background elements (distant trees, buildings, hills) are CONTEXT only and must never override the dominant foreground cover.

Carefully observe and describe the following components in a single, detailed, and cohesive paragraph:
1. image_medium: State clearly whether this is a real-world photograph, a painting/artwork, a map, an abstract drawing/illustration, a text graphic, or a close-up indoor sign/object.
2. dominant_cover: The single land-cover type occupying most of the foreground/center (e.g. cereal crop, grassland, bare soil, broadleaved trees, built surface, sand, snow/ice, water, wetland).
3. cover_fraction_estimate: Share of the visible ground area that is vegetation vs bare soil/rock vs sealed/built surface vs water/ice.
4. visible_evidence & salient_objects: Primary foreground objects, structures, architectural elements, vehicles, or natural formations visible. Describe them naturally as a human observer would.
5. vegetation_detail: Height, leaf/frond shape, seed-head/flower form, rows or not (cereals, broadleaf/industrial crops, grassland, woody cover broadleaved vs coniferous vs palm).
6. vegetation_condition & phenological_stage: Vigor (vigorous-green, senescent, dead/brown) and growth stage (emerging, full-canopy, harvested/stubble).
7. canopy_structure: Canopy closure (closed, open, scattered) and height (low shrub <2m, small trees 2-5m, mature trees >5m).
8. soil_surface_state & lithology: Exposed ground sealing (permeable, unsealed, sealed-impervious) and visibly exposed soil/rock texture and color.
9. structure_and_pattern: Terrain slope, parcel boundaries, linear features (crop rows, tramlines, ditches, tracks), or recent disturbances.
10. context_background: Elements visible in the distance or periphery (buildings, treelines, mountains, water bodies) that are not the sample point cover.
11. human_evidence & activities: Visible signs of human use (buildings, machinery, fences, planted rows, paths, irrigation, or a selfie where a person is visiting a landmark or hiking).

Do not assign a LULC category code or label. Focus purely on describing what is visually supported.
```

---

### 🗺️ Step 2: Geographical LULC Classification (Text-Only)
* **Input**: The visual description output from Step 1 + geographic/climatic metadata + the LULC category list + `prompts/shared/prompt_step2.txt`
* **Output**: The final structured classification label and a consolidated description paragraph.

**Prompt Template (`prompts/shared/prompt_step2.txt`):**
```text
You are a geographical and ecological classifier. Your task is to classify the land cover/land use of a location based on a visual description of the scene and geographic/climatic metadata.

[METADATA CONTEXT]
Location coordinates/bounding box: {location}
Region/Country: {country}
Continent: {continent}
Time of Day: {time_of_day}
Season: {season}
Köppen-Geiger Climate Code: {koppen_code} ({koppen_desc})

[VISUAL DESCRIPTION]
{visual_description}

[CLASSIFICATION RULES]
1. Use the climate code and season to resolve ambiguities (e.g. distinguishing tropical savanna from temperate grassland, or identifying agricultural crops vs natural vegetation).
2. If the image medium is noted as a painting, artwork, illustration, drawing, sketch, map, diagram, close-up text graphic, or indoor sign, classify it as "None of the above / Noise".
3. Selfies or tourist photos of people are fully acceptable as long as the background is visible; classify the scene based on the background cover and context.
4. If any metadata attributes (such as Season, Time of Day, or Country/Continent) are specified as "Unknown", rely primarily on the visual description and other available metadata elements.

Based on the metadata and visual description, classify this environment into EXACTLY one of the following LULC categories:
{lulc_list}

Format your output EXACTLY as follows:
LABEL: <Insert EXACTLY one category from the list above>
DESCRIPTION: <A detailed, cohesive paragraph in fluent natural language describing the visual evidence, human activities, land cover, and vegetation. This paragraph must integrate all of the observed components and metadata naturally and concisely.>
```

---

## 🗂️ Land Use / Land Cover (LULC) Classification Vocabulary

### Natural LULC Categories
* **Broadleaved forest**: Deciduous or evergreen broad-leaf trees (oak, beech, maple, birch).
* **Coniferous forest**: Evergreen needle-leaf trees (pine, spruce, fir, larch).
* **Mixed forest**: Co-dominant broadleaved and coniferous trees.
* **Tropical forest**: Equatorial rainforests, mangroves, or tropical dry forest.
* **Sparsely wooded / Savanna**: Grassland with scattered trees (10-30% canopy cover).
* **Natural grassland**: Meadows, wild steppes, alpine grasslands, or prairies.
* **Temperate shrubland / Scrub**: Low woody scrub (heather, gorse, bramble).
* **Arid shrubland**: Desert scrub, sagebrush, or dry savanna bushland.
* **Tundra**: Low-growing polar vegetation (mosses, lichens, dwarf shrubs).
* **Sandy desert / Dunes**: Sand sheets, active dunes, or sandy flats.
* **Rocky desert / Gravel plains**: Stony hamadas, gravel plains, or barren volcanic ash fields.
* **Barren soil / Badlands**: Highly eroded clay hills, bare dry earth, or dry salt flats.
* **Bare rock / Cliffs**: Exposed bedrock, cliffs, or scree slopes.
* **Mountain peak / Alpine ridge**: High-altitude rocky ridges, jagged summits, or mountain peaks.
* **Waterfall / Cascade**: Water flowing vertically over a cliff or steep drop.
* **Volcanic terrain / Lava flow**: Barren basalt fields, volcanic craters, or active geothermal areas.
* **Salt flat / Playa**: Dry lake beds covered with salt crusts or mineral deposits.
* **Coastal beach / Spit**: Sandy or pebbly sea coast.
* **Wetland / Marsh / Bog**: Marshes, peat bogs, fens, reed beds, or swamps.
* **River / Stream**: Flowing freshwater channels, creeks, or canals.
* **Lake / Pond**: Standing inland water bodies or reservoirs.
* **Marine / Estuary**: Coastal saltwater, ocean surf, bays, or intertidal flats.
* **Glacier / Permanent ice**: Glaciers, ice caps, or permanent snowfields.
* **Other natural land cover**: Any other natural land cover or landscape.

### Man-made LULC Categories
* **Forest plantation**: Planted timber rows (e.g. eucalyptus, pine crop).
* **Managed pasture**: Fenced grazing land or paddocks.
* **Herbaceous cropland**: Annual cultivated crops (wheat, corn, barley, canola).
* **Orchards & Vineyards**: Perennial woody crops (fruit orchards, grapevines).
* **Rice paddies / Flooded crops**: Water-flooded agricultural basins.
* **Covered agriculture**: Greenhouses, polytunnels, or nurseries.
* **High-density built-up**: Urban centers, skyscrapers, and high-rise blocks.
* **Suburban / Low-density residential**: Single-family homes, private yards, villas, and neighborhood streets.
* **Industrial / Commercial zone**: Factories, warehouses, oil refineries, or retail parks.
* **Active construction site**: Earthworks, building foundations, cranes, and scaffolding.
* **Transportation network**: Highways, railways, runways, or shipping ports.
* **Mine / Quarry / Landfill**: Open-pit mines, stone quarries, or landfill sites.
* **Urban green space**: City parks, golf courses, botanical gardens, or sports fields.
* **Historical / Cultural monument / Archaeological site**: Ancient ruins, historic temples, or archeological landmarks.
* **Other man-made surface**: Any other artificial or managed land cover or surface.

---

## 🩹 Fallback Retry Safety (`relabel_failed_clusters.py`)

Due to HTTP timeouts or server errors, certain clusters may fail MLLM processing (labeled as `"Error Labeling"` or `"Unlabeled"`).
* The script scans the metadata database and identifies failed clusters.
* For each failed cluster, it performs **depth retries (up to `--fallback_depth 20`)**, pulling sequentially further representative images in the cluster card and re-running the classification.
* This guarantees $100\%$ labeling coverage of all visual nodes.
