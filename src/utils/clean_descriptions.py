"""Clean layout and collage artifacts from cluster descriptions.

Removes meta-language referring to image presentation formats (e.g. 'vertical stack',
'across the four frames', 'in the bottom frame', 'composite image') from visual
and cluster descriptions, replacing them with natural phrasing describing the scene directly.
"""

import argparse
import logging
import os
import re
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def clean_layout_artifacts(text: Optional[str]) -> Optional[str]:
    """Scrub layout and presentation-format artifacts from a description.

    Transforms references to image layout (e.g. vertical stacks, frames, collages)
    into direct references to the physical environment and representative views.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return text

    original = text

    # --- 1. Standalone Opening Sentences ---
    # "The input image is a vertical stack of four real-world photographs."
    text = re.sub(
        r"^The\s+input\s+image\s+is\s+a\s+(?:vertical\s+stack|composite\s+image|composite|montage|collage)\s+of\s+\w+\s+(?:real-world\s+)?(?:photographs|photos|images)\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # "The input image is a real-world photograph, presented as a vertical stack of four distinct frames."
    text = re.sub(
        r"^The\s+input\s+image\s+is\s+a\s+(?:real-world\s+)?photograph,\s+presented\s+as\s+a\s+(?:vertical\s+stack|composite\s+image|composite|montage|collage)[^.]*\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # "The input image contains a vertical stack of four representative photographs from the same local cluster."
    text = re.sub(
        r"^The\s+input\s+image\s+contains\s+a\s+(?:vertical\s+stack|composite\s+image|composite|montage|collage)[^.]*\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # --- 2. Opening Clauses with verbs (depicting, showing, appearing, etc.) ---
    # "The input image is a vertical stack of four real-world photographs, all appearing to be taken..."
    text = re.sub(
        r"^The\s+input\s+image\s+is\s+a\s+(?:vertical\s+stack|composite\s+image|composite|montage|collage)\s+of\s+\w+\s+(?:real-world\s+)?(?:photographs|photos|images),\s*(?:all|each)?\s*(?:appearing\s+to\s+be|appear\s+to\s+be)\s*",
        "Representative views appear to be ",
        text,
        flags=re.IGNORECASE,
    )
    # "The input image is a vertical stack of four real-world photographs, all depicting..."
    text = re.sub(
        r"^The\s+input\s+image\s+is\s+a\s+(?:vertical\s+stack|composite\s+image|composite|montage|collage)\s+of\s+\w+\s+(?:real-world\s+)?(?:photographs|photos|images),\s*(?:all|each)?\s*(?:depicting|showing)\s*",
        "Representative views depict ",
        text,
        flags=re.IGNORECASE,
    )
    # "The input image is a vertical stack of four real-world photographs, all/each..."
    text = re.sub(
        r"^The\s+input\s+image\s+is\s+a\s+(?:vertical\s+stack|composite\s+image|composite|montage|collage)\s+of\s+\w+\s+(?:real-world\s+)?(?:photographs|photos|images),\s*(?:all|each)?\s*",
        "Representative views show ",
        text,
        flags=re.IGNORECASE,
    )

    # --- 3. Stacked Photographs / Vertical Stack Phrases ---
    # "across the four stacked photographs" -> "across the scene"
    text = re.sub(
        r"\bacross\s+the\s+(?:\w+\s+)?(?:stacked\s+photographs|stacked\s+photos)\b",
        "across the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bin\s+the\s+(?:\w+\s+)?(?:stacked\s+photographs|stacked\s+photos)\b",
        "in the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:across|in|from|throughout)\s+the\s+vertical\s+stack(?:\s+of\s+(?:photographs|photos|images))?\b",
        "across the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bthe\s+vertical\s+stack\s+of\s+(?:photographs|images|photos)\b",
        "the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\ba\s+vertical\s+stack\s+of\s+\w+\s+(?:photographs|images|photos)\b",
        "representative views of the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bvertical\s+stack\s+of\s+(?:photographs|images|photos)\b",
        "representative views of the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bvertical\s+stack\b", "scene", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:the\s+)?stacked\s+photographs\b",
        "the photographs",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bstacked\s+photos\b", "photographs", text, flags=re.IGNORECASE)

    # "a stack of four photographs" / "stack of images"
    text = re.sub(
        r"\b(?:a\s+)?stack\s+of\s+(?:\w+\s+)?(?:photographs|photos|images)\b",
        "representative views of the scene",
        text,
        flags=re.IGNORECASE,
    )

    # --- 4. Frame References ---
    # "of all frames" -> "of the scene"
    text = re.sub(
        r"\bof\s+all\s+(?:the\s+)?frames\b",
        "of the scene",
        text,
        flags=re.IGNORECASE,
    )
    # "across the four frames", "across all frames", "across the multiple frames"
    text = re.sub(
        r"\bacross\s+(?:the\s+)?(?:all|\w+)\s+(?:distinct\s+)?frames\b",
        "across the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bacross\s+(?:the\s+)?(?:all\s+of\s+the\s+)?frames\b",
        "across the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bin\s+all\s+(?:the\s+)?frames\b",
        "in all views",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bbetween\s+the\s+(?:distinct\s+)?frames\b",
        "across the views",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bin\s+each\s+frame\b",
        "in each view",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:across|throughout)\s+the\s+frames\b",
        "across the views",
        text,
        flags=re.IGNORECASE,
    )
    # "in the lower/upper/bottom/top/middle/second/third/fourth frame"
    text = re.sub(
        r"\bin\s+the\s+(?:lower|upper|bottom|top|middle|first|second|third|fourth)\s+frame\b",
        "in specific views",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:The\s+)?(?:lower|upper|bottom|top|middle|first|second|third|fourth)\s+frame\b",
        "A specific vantage point",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:The\s+)?(?:lower|upper|bottom|top|middle)\s+(?:\w+\s+)?frames\b",
        "Specific vantage points",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bindividual\s+frames\b",
        "individual views",
        text,
        flags=re.IGNORECASE,
    )

    # --- 5. Composite Image References ---
    text = re.sub(
        r"\b(?:the\s+)?composite\s+image\b",
        "the scene",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\ba\s+composite\s+image\b",
        "a visual scene",
        text,
        flags=re.IGNORECASE,
    )

    # --- 6. Clean whitespace, punctuation and capitalization ---
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Fix trailing spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Fix empty sentence remnants like ". The" at start
    text = re.sub(r"^[.,;:\s]+", "", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def clean_parquet_file(
    input_parquet: str, output_parquet: Optional[str] = None
) -> None:
    """Cleans description columns in a parquet file atomically and in-place."""
    if output_parquet is None:
        output_parquet = input_parquet

    logger.info("Reading %s...", input_parquet)
    df = pd.read_parquet(input_parquet)

    target_cols = [
        "visual_description",
        "cluster_description",
        "parent_visual_description",
        "parent_cluster_description",
    ]
    present_cols = [c for c in target_cols if c in df.columns]

    if not present_cols:
        logger.warning("No description columns found in %s. Skipping.", input_parquet)
        return

    for col in present_cols:
        logger.info("Cleaning column '%s'...", col)
        unique_vals = df[col].dropna().unique()
        clean_map = {val: clean_layout_artifacts(val) for val in unique_vals}
        df[col] = df[col].map(clean_map).fillna(df[col])

    # Atomic write
    dir_name = os.path.dirname(os.path.abspath(output_parquet))
    base_name = os.path.basename(output_parquet)
    tmp_path = os.path.join(dir_name, f".tmp_clean_{base_name}")

    logger.info("Writing cleaned dataset to %s...", tmp_path)
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_parquet)
    logger.info("Atomically updated %s successfully.", output_parquet)


def main():
    parser = argparse.ArgumentParser(
        description="Clean layout and presentation artifacts from clustered Parquet files."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to clustered Parquet file to clean.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path for output file. If omitted, cleans in-place.",
    )
    args = parser.parse_args()

    clean_parquet_file(args.parquet, args.out)


if __name__ == "__main__":
    main()
