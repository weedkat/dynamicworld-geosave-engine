"""Ingest Sentinel-2 imagery + remapped DynamicWorld labels for every raw label tile.

Each raw .tif is both the anchor (location + datetime, from its own
``-YYYYMMDD`` filename suffix) *and* the label source itself — Pipeline
fetches matching Sentinel-2 imagery via STAC for the anchor, and this script
separately remaps the anchor's own raw pixels into the ``dynamicworld``
layer. Label remapping is specific to this exact raw data release's class
encoding (see ../data/dynamicworld_raw/README.txt), so it lives here, not in
the general-purpose Pipeline.

Imagery and label are two fully independent steps, not bundled into one
Pipeline: Pipeline.ingest() builds the imagery layers untouched, and this
script attaches the label to each yielded sample (flattening the pipeline's
own GeoStack positionally into a new one alongside it, via GeoStack's
constructor) before the one save — no separate pass, no reload. class_map/color_map for the
*remapped* classes live in configs/metadata.yaml
(read by SemanticSegmentationTask at train time), not here — nothing in the
training path reads GeoTile.metadata, so duplicating them into the saved
tile would just be a second, driftable copy of the same information.

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

from modules.data_pipeline import Pipeline

log = logging.getLogger(__name__)
load_dotenv()

RAW_ROOT = Path("data/dynamicworld_raw")
RAW_DIRS = {split: RAW_ROOT / split for split in ("train", "val", "test")}
OUT_ROOT = Path("data/dynamicworld")

LABEL_BAND_NAME = "label"
LABEL_NODATA = 255  # matches configs/metadata.yaml's ignore_index

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


def build_label(anchor: GeoTile) -> GeoTile:
    """Remap the anchor's own raw pixel values into the target label schema."""
    label = remap(anchor, LABEL_REMAP)
    da = label.data.isel(band=[0])  # select the first band (assuming single-band label)
    label = label.with_data(da.assign_coords(band=["label"]))
    return label.with_nodata(LABEL_NODATA).with_metadata({"description": "Remapped DynamicWorld labels"})


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
    for group_dir in tqdm(find_groups(raw_dir), desc=f"Ingesting {raw_dir.name}", unit="group"):
        anchor_paths = [p for p in sorted(group_dir.glob("*.tif"))]
        root = out_root / group_dir.relative_to(raw_dir)
        pipeline = Pipeline()

        for anchor_path in tqdm(anchor_paths, desc=f"Ingesting {group_dir.name}", unit="anchor", leave=False):
            anchor = GeoTile.from_geotiff(anchor_path, load_data=True)
            geostack_dir = root / f"{anchor.stem}.geostack"
            if geostack_dir.exists():
                continue  # already ingested
            
            stack = next(pipeline.ingest(anchor))
            try:
                GeoStack(stack, dynamicworld=build_label(anchor)).save(
                    geostack_dir, save_stac=["sentinel_2_l1c"]
                )
            except Exception as e:
                log.error("Failed to save a sample for %s: %s", anchor.stem, e)


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
