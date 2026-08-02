"""Ingest Sentinel-2 imagery + remapped DynamicWorld labels for every raw label tile.

Each raw .tif is both the anchor (location + datetime, from its own
``-YYYYMMDD`` filename suffix) *and* the label source itself — Pipeline
fetches matching Sentinel-2 imagery via STAC for the anchor, and this script
separately remaps the anchor's own raw pixels into a ``dynamicworld``
layer, keeping the untouched original as ``dynamicworld_raw`` alongside it
(GeoStack's zarr groups make both cheap to keep). Label remapping is
specific to this exact raw data release's class encoding (see
../data/dynamicworld_raw/README.txt), so it lives here, not in the
general-purpose Pipeline.

Imagery and label are two fully independent steps, not bundled into one
Pipeline: Pipeline.ingest() builds the imagery layers untouched, and this
script attaches both label layers to each yielded sample (flattening the
pipeline's own GeoStack positionally into a new one alongside them, via
GeoStack's constructor) before the one save — no separate pass, no reload.
LABEL_CLASS_MAP/LABEL_COLOR_MAP mirror configs/metadata.yaml's
model.init_args.class_map/color_map by hand (kept inline, not read from
that file, so this script stays self-contained) — only used for
plot_meta so a saved sample renders readably via .plot(); nothing in the
training path touches plot_meta.

Output is grouped to mirror the raw data's own folder structure (e.g.
train/Experts/EH/1/, train/Non_expert/WorkForce/EH/1/), whatever that
nesting happens to be per split — not hardcoded here. README/metadata files
(not raster data) are copied alongside the ingested output too, same
relative layout, so the dataset carries its own provenance docs.
"""
import logging
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from geosave_engine.geodata.tile import GeoStack, GeoTile, remap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # allow imports from workspace/ without installing the package

from modules.data_pipeline_l2a import Pipeline

log = logging.getLogger(__name__)
load_dotenv()

RAW_ROOT = Path("data/dynamicworld_raw")
RAW_DIRS = {
    "train": RAW_ROOT / "train",
    "val": RAW_ROOT / "val",
    "test": RAW_ROOT / "test"
}
OUT_ROOT = Path("data/dynamicworld")

LABEL_NODATA = 255  # matches configs/metadata.yaml's ignore_index

# https://doi.pangaea.de/10.1594/PANGAEA.933475
DW_CLASS_MAP = { 
    0: "No Data",
    1: "Water",
    2: "Trees",
    3: "Grass",
    4: "Flooded Vegetation",
    5: "Crops",
    6: "Scrub",
    7: "Built Area",
    8: "Bare Ground",
    9: "Snow and Ice",
    10: "Cloud",
}

DW_COLOR_MAP = {
    0: "#000000",
    1: "#419bdf",
    2: "#397d49",
    3: "#88b053",
    4: "#7a87c6",
    5: "#e49635",
    6: "#dfc35a",
    7: "#c4281b",
    8: "#a59b8f",
    9: "#f0f0f0",
    10: "#d3d3d3"
}

# Remapped 0..7 schema (mirrors configs/metadata.yaml's model.init_args.class_map/color_map —
# keep both in sync by hand if that file changes).
LABEL_CLASS_MAP = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded Vegetation",
    4: "Crops",
    5: "Scrub",
    6: "Built Area",
    7: "Bare Ground",
    255: "Ignore",
}

LABEL_COLOR_MAP = {
    0: "#419bdf",
    1: "#397d49",
    2: "#88b053",
    3: "#7a87c6",
    4: "#e49635",
    5: "#dfc35a",
    6: "#c4281b",
    7: "#a59b8f",
    255: "#000000",
}

# Raw Tier 1 class values (README.txt) -> remapped 0..7 schema (configs/metadata.yaml's
# class_map). Safe to apply sequentially: each dst value equals an already-processed
# src value, never one still pending, so earlier writes never get clobbered by later ones.
LABEL_REMAP = {
    0: LABEL_NODATA,   # no data
    1: 0,              # water
    2: 1,              # trees
    3: 2,              # grass
    4: 3,              # flooded vegetation
    5: 4,              # crops
    6: 5,              # scrub
    7: 6,              # built area
    8: 7,              # bare ground
    9: LABEL_NODATA,   # snow/ice -> ignore
    10: LABEL_NODATA,  # cloud -> ignore
}


def build_label(anchor: GeoTile) -> dict[str, GeoTile]:
    """Raw + remapped DynamicWorld label, as two GeoStack layers sharing one anchor.

    Single-band label, so both squeeze straight to (y, x) — no band name needed.

    Returns:
        {"dynamicworld_raw": <untouched Tier 1 values>, "dynamicworld": <remapped 0..7>}
    """
    raw = anchor.rebase(
        data=anchor.data.isel(band=0),
        metadata={"description": "Raw DynamicWorld labels (Tier 1 raw class values)"},
        plot_meta={"class_map": DW_CLASS_MAP, "color_map": DW_COLOR_MAP},
    )
    remapped = remap(anchor, LABEL_REMAP)
    remapped = remapped.rebase(
        data=remapped.data.isel(band=0),
        nodata=LABEL_NODATA,
        metadata={"description": "Remapped DynamicWorld labels"},
        plot_meta={"class_map": LABEL_CLASS_MAP, "color_map": LABEL_COLOR_MAP},
    )
    return {"dynamicworld_raw": raw, "dynamicworld": remapped}


def find_groups(raw_dir: Path) -> list[Path]:
    """Every directory under raw_dir that directly contains .tif files.

    One cheap pass — just filesystem metadata (rglob + path.parent), no
    anchors loaded — so scoping out the whole raw tree's structure up front
    costs nothing worth avoiding. Anchor loading is the actually heavy step
    (GDAL open per file); that stays deferred until `ingest_split`'s own
    per-group loop.

    Args:
        raw_dir: Raw split root (e.g. data/dynamicworld_raw/train).

    Returns:
        Sorted leaf directories (e.g. train/Experts/EH/1/) with .tif files.
    """
    return sorted({p.parent for p in raw_dir.rglob("*.tif")})


def ingest_split(raw_dir: Path, out_root: Path) -> None:
    """Ingest every raw .tif under raw_dir, grouped by its own leaf directory.

    Args:
        raw_dir: Raw split root (e.g. data/dynamicworld_raw/train).
        out_root: Output split root (e.g. data/dynamicworld/train) — each
            group's anchors save to out_root/<same relative subfolder>.
    """
    pipeline = Pipeline()
    
    for group_dir in tqdm(find_groups(raw_dir), desc=f"Ingesting {raw_dir.name}", unit="group"):
        anchor_paths = [p for p in sorted(group_dir.glob("*.tif"))]
        root = out_root / group_dir.relative_to(raw_dir)

        for anchor_path in tqdm(anchor_paths, desc=f"Ingesting {group_dir.name}", unit="anchor", leave=False):
            anchor = GeoTile.from_geotiff(anchor_path, load_data=True)
            geostack_dir = root / f"{anchor.stem}.geostack"
            if geostack_dir.exists():
                continue  # already ingested
            try:
                stack = next(pipeline.ingest(anchor.to_anchor()))
                GeoStack(stack, **build_label(anchor)).save(geostack_dir)
            except Exception as e:
                log.error("Failed to ingest/save a sample for %s: %s", anchor.stem, e)


def copy_docs(raw_root: Path, out_root: Path) -> None:
    """Copy README/metadata files (not raster data) alongside the ingested output."""
    for pattern in ("*.txt", "*.xlsx"):
        for src in raw_root.rglob(pattern):
            dst = out_root / src.relative_to(raw_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


if __name__ == "__main__":
    for split, raw_dir in RAW_DIRS.items():
        ingest_split(raw_dir, OUT_ROOT / split)
    copy_docs(RAW_ROOT, OUT_ROOT)
