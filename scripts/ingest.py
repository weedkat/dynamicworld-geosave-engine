"""Ingest Sentinel-2 imagery + remapped DynamicWorld labels for every raw label tile.

Each raw .tif is both the anchor (location + datetime, from its own
``-YYYYMMDD`` filename suffix) *and* the label source itself — Pipeline
fetches matching Sentinel-2 imagery via STAC for the anchor, and this script
separately remaps the anchor's own raw pixels into the ``dynamicworld``
layer. Label remapping is specific to this exact raw data release's class
encoding (see ../data/dynamicworld_raw/README.txt), so it lives here, not in
the general-purpose Pipeline.

Output is grouped to mirror the raw data's own folder structure (e.g.
train/Experts/EH/1/, train/Non_expert/WorkForce/EH/1/), whatever that
nesting happens to be per split — not hardcoded here. README/metadata
files (not raster data) are copied alongside the ingested output too, same
relative layout, so the dataset carries its own provenance docs.
"""
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

from geosave_engine.geodata.pipeline import save_dataset
from geosave_engine.geodata.tile import GeoTile, remap
from modules.data_pipeline import Pipeline

load_dotenv()

RAW_ROOT = Path("data/dynamicworld_raw")
RAW_DIRS = {split: RAW_ROOT / split for split in ("train", "val", "test")}
OUT_ROOT = Path("data/dynamicworld")

LABEL_SPEC = yaml.safe_load(Path("modules/label.yaml").read_text())
LABEL_CLASS_MAP = {s["id"]: s["name"] for s in LABEL_SPEC["schema"]}
LABEL_COLOR_MAP = {s["id"]: s["color"] for s in LABEL_SPEC["schema"]}
LABEL_BAND_NAME = "label"

# Raw Tier 1 class values (README.txt) -> label.yaml's remapped schema.
# Safe to apply sequentially (each dst_val equals an already-processed
# src_val, never a src_val still pending), so earlier writes never get
# clobbered by later steps.
LABEL_REMAP = {
    0: LABEL_SPEC["nodata"],   # no data (left unmarked)
    1: 0,                      # water
    2: 1,                      # trees
    3: 2,                      # grass
    4: 3,                      # flooded vegetation
    5: 4,                      # crops
    6: 5,                      # scrub
    7: 6,                      # built area
    8: 7,                      # bare ground
    9: LABEL_SPEC["nodata"],  # snow/ice -> ignore
    10: LABEL_SPEC["nodata"], # cloud -> ignore
}


def build_label(anchor: GeoTile) -> GeoTile:
    """Remap the anchor's own raw pixel values into label.yaml's schema."""
    label = remap(anchor, LABEL_REMAP)
    label = label.with_data(label.data.assign_coords(band=[LABEL_BAND_NAME]))
    return label.with_nodata(LABEL_SPEC["nodata"]).with_metadata({
        "description": LABEL_SPEC["description"],
        "class_map": LABEL_CLASS_MAP,
        "color_map": LABEL_COLOR_MAP,
    })


def ingest_split(raw_dir: Path, out_root: Path) -> None:
    """Ingest every raw .tif under raw_dir, grouped by its own relative subfolder.

    Two independent steps per group: save_dataset writes the imagery layers
    (Pipeline, untouched); then the label is written straight into each
    anchor's already-existing .geostack folder, as its own dynamicworld.zarr
    — no need to route it through Pipeline.ingest() at all.

    Args:
        raw_dir: Raw split root (e.g. data/dynamicworld_raw/train).
        out_root: Output split root (e.g. data/dynamicworld/train) — each
            group's anchors save to out_root/<same relative subfolder>.
    """
    groups: dict[Path, list[GeoTile]] = defaultdict(list)
    for tiff_path in raw_dir.rglob("*.tif"):
        rel_group = tiff_path.relative_to(raw_dir).parent
        groups[rel_group].append(GeoTile.from_geotiff(tiff_path))

    for rel_group, anchors in groups.items():
        root = out_root / rel_group
        save_dataset(Pipeline(), anchors, root)

        for anchor in anchors:
            geostack_dir = root / f"{anchor.stem}.geostack"
            label_path = geostack_dir / "dynamicworld.zarr"
            if not geostack_dir.exists() or label_path.exists():
                continue  # imagery ingest failed, or label already written
            build_label(anchor).to_zarr(label_path)


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
