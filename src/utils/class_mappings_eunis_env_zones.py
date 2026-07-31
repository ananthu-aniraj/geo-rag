# Centralized class mappings for EUNIS Ecosystems and Environmental Zones of Europe

# Standard EUNIS Ecosystem Class/MAES Mapping
# Maps raw raster code values (1-10) or level-1 alphabet characters to MAES Level 2 / EUNIS classifications
EUNIS_ECOSYSTEM_MAPPING = {
    # MAES Level 1 / EUNIS Level 1 Integer codes
    1: "Urban / Artificial",
    2: "Cropland / Agricultural",
    3: "Grassland",
    4: "Woodland and forest",
    5: "Heathland and shrub",
    6: "Sparsely vegetated land",
    7: "Wetland / Mire / Bog",
    8: "Rivers and lakes (Inland water)",
    9: "Marine / Sea",
    10: "Coastal / Dunes",

    # EUNIS Level 1 Alphabet codes
    "A": "Marine habitats",
    "B": "Coastal habitats",
    "C": "Inland surface waters",
    "D": "Mires, bogs and fens",
    "E": "Grasslands",
    "F": "Heathland, scrub and tundra",
    "G": "Woodland, forest and other wooded land",
    "H": "Inland habitats with very sparse or no vegetation",
    "I": "Cultivated agricultural, horticultural and domestic habitats",
    "J": "Constructed, industrial and other artificial habitats"
}

# Environmental Zones 2025 (version 2.0 / Metzger 2025) Value Mapping (1-19)
ENV_ZONES_MAPPING = {
    1: "Alpine North (ALN)",
    2: "Boreal (BOR)",
    3: "Nemoral (NEM)",
    4: "Atlantic North (ATN)",
    5: "Atlantic Central (ATC)",
    6: "Lusitanian (LUS)",
    7: "Alpine South (ALS)",
    8: "Continental (CON)",
    9: "Pannonian (PAN)",
    10: "Mediterranean North (MDN)",
    11: "Mediterranean Mountains (MDM)",
    12: "Mediterranean South (MDS)",
    13: "Aegean (AEG)",
    14: "Blacksea climate region (BSC)",
    15: "Central Anatolian (CAN)",
    16: "Eastern Anatolian (EAN)",
    17: "Southwest Anatolian transition region (SAN)",
    18: "Macaronesian (MAC)",
    19: "Arctic (ARC)"
}
