from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy.ndimage import binary_opening

from geosave_engine.geodata.tile import GeoTile
from geosave_engine.geodata.features import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_ndvi,
    compute_s2c_mask,
)
from geosave_engine.geodata.pipeline import GeoPipeline, SourceProtocol
from geosave_engine.geodata.stac import StacClient

L1C_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B09", "B10", "B11", "B12", "B8A",
]

# DynamicWorld paper's model input set: all bands except B01/B8A/B09/B10 —
# fetched anyway (L1C_BANDS above) since cloud-mask derivation needs them,
# but the saved sentinel_2_l1c layer only carries what the model actually
# takes. odc-stac already resamples every band onto one geobox (bilinear by
# default, matching the paper) at StacSource.load time — no separate
# upscaling step needed here.
DW_MODEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]


class Pipeline(GeoPipeline):
    """Sentinel-2 imagery + cloud/shadow mask + NDVI for one anchor.

    Imagery only — DynamicWorld label prep is a separate, project-specific
    step (see ``workspace/scripts/dynamic_world_ingest``), writing
    ``dynamicworld.zarr`` directly into each anchor folder this pipeline's
    ``save_dataset()`` run already created. Not this class's concern: label
    remapping doesn't generalize across projects the way STAC-driven
    imagery ingest does.
    """

    @cached_property
    def sources(self) -> dict[str, SourceProtocol]:
        """Built lazily on first real fetch — importing/instantiating Pipeline
        for .context() alone must not cost a live STAC network call."""
        stac_client = StacClient.cdse()
        return {"sentinel_2_l1c": stac_client.source("sentinel-2-l1c", bands=L1C_BANDS, max_nodata_fraction=0.1)}

    def context(self, tiles: dict[str, GeoTile]) -> dict[str, object]:
        ref = next(iter(tiles.values()))
        return {
            "crs": ref.crs,
            "transform": ref.affine,
            "coordinate": ref.centroid,
            "time": ref.start.timetuple().tm_yday,
            "datetime": ref.start.isoformat(),
            "bbox_wgs84": list(ref.wgs84_bbox),
            "stac_item_ids": [i.id for i in ref.stac],
        }

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        s2 = raw["sentinel_2_l1c"]
        # Cloud mask/NDVI derive from the full 13-band fetch (they need
        # B01/B8A/B09/B10) — select down to the model's own input bands
        # only after they're computed.
        cloud_mask = self._ingest_cloud_mask(s2)
        ndvi = self._ingest_ndvi(s2)
        s2_model = s2.with_data(s2.data.sel(band=DW_MODEL_BANDS))
        # Tag descriptions last: cloud_mask/ndvi derive from s2 via with_np, which
        # carries s2's metadata along — tagging s2 first would leak its description
        # into both, then clash with theirs.
        return {
            "sentinel_2_l1c": s2_model.with_metadata(
                {"description": f"Sentinel-2 L1C imagery ({len(DW_MODEL_BANDS)} bands, DynamicWorld input set)"}
            ),
            "cloud_mask": cloud_mask.with_metadata({"description": "Cloud and shadow mask, 0=clear, 1=cloud/shadow"}),
            "ndvi": ndvi.with_metadata({"description": "Normalized Difference Vegetation Index"}),
        }

    def _mask_one(self, ds, sun_az: float, resolution: float) -> np.ndarray:
        """Cloud + shadow mask for one time slice. `ds` has dims (band, y, x)."""
        s2c = compute_s2c_mask(
            b01=ds.sel(band="B01").values, b02=ds.sel(band="B02").values,
            b04=ds.sel(band="B04").values, b05=ds.sel(band="B05").values,
            b08=ds.sel(band="B08").values, b8a=ds.sel(band="B8A").values,
            b09=ds.sel(band="B09").values, b10=ds.sel(band="B10").values,
            b11=ds.sel(band="B11").values, b12=ds.sel(band="B12").values,
            prob_threshold=0.4,
        )
        cdi = compute_cdi_mask(
            b07=ds.sel(band="B07").values,
            b08=ds.sel(band="B08").values,
            b8a=ds.sel(band="B8A").values,
        )
        cirrus = compute_b10_mask(b10=ds.sel(band="B10").values, b10_threshold=0.0012)
        cloud_mask = binary_opening(s2c & cdi & cirrus, structure=np.ones((3, 3)))
        shadow_mask = build_shadow_mask(cloud_mask, sun_azimuth_deg=sun_az, resolution=int(resolution))
        return (cloud_mask | shadow_mask).astype(np.uint8)

    def _ingest_cloud_mask(self, s2: GeoTile) -> GeoTile:
        ds = s2.data
        # Sun azimuth varies slowly scene-to-scene for one location/season —
        # using the first matched item's for every time step is a fine approximation.
        sun_az = s2.stac[0].properties.get("view:sun_azimuth", 0.0)

        if s2.has_time:
            masks = [self._mask_one(ds.sel(time=t), sun_az, s2.resolution) for t in ds.time.values]
            stacked = np.stack(masks)[:, np.newaxis]  # (time, band=1, y, x)
            return s2.with_np(stacked, ["cloud_mask"], times=list(s2.times))

        mask = self._mask_one(ds, sun_az, s2.resolution)
        return s2.with_np(mask, ["cloud_mask"])

    def _ingest_ndvi(self, s2: GeoTile) -> GeoTile:
        ds = s2.data
        ndvi = compute_ndvi(nir=ds.sel(band="B08").values, red=ds.sel(band="B04").values)
        if s2.has_time:
            stacked = ndvi[:, np.newaxis].astype(np.float32)  # (time, band=1, y, x)
            return s2.with_np(stacked, ["ndvi"], times=list(s2.times))
        return s2.with_np(ndvi.astype(np.float32), ["ndvi"])
