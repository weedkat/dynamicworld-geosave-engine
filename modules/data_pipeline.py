from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_opening

from geosave_engine.geodata.tile import GeoTile
from geosave_engine.geodata.features import (
    build_shadow_mask,
    compute_b10_mask,
    compute_cdi_mask,
    compute_ndvi,
    compute_s2c_mask,
)
from geosave_engine.geodata.pipeline import GeoPipeline
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.stac.source import StacSource

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
    step (see ``workspace/scripts/ingest.py``), attached to each sample this
    pipeline's ``ingest()`` yields before that script's own single save. Not
    this class's concern: label remapping doesn't generalize across
    projects the way STAC-driven imagery ingest does.
    """
    @cached_property
    def sources(self) -> dict[str, StacSource]:
        """Built lazily on first real fetch — importing/instantiating Pipeline
        alone must not cost a live STAC network call."""
        stac_client = StacClient.cdse()
        # temporal_slots=1 (scene granularity, StacSource default) — preprocess()
        # below assumes exactly one time step per raw sample, no loop.
        return {
            "sentinel_2_l1c": stac_client.source(
                "sentinel-2-l1c", bands=L1C_BANDS, max_nodata_fraction=0.1, temporal_slots=1
            )
        }

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        s2 = raw["sentinel_2_l1c"]
        # temporal_slots=1 on the source (see `sources` above) — exactly one
        # scene per sample, so drop straight to (band, y, x), no time loop.
        ds = s2.data.isel(time=0)
        sun_az = s2.stac[0].properties.get("view:sun_azimuth", 0.0)

        # Cloud mask/NDVI derive from the full 13-band fetch (they need
        # B01/B8A/B09/B10) — select down to the model's own input bands
        # only after they're computed.
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
        shadow_mask = build_shadow_mask(cloud_mask, sun_azimuth_deg=sun_az, resolution=int(s2.resolution))
        mask = (cloud_mask | shadow_mask).astype(np.uint8)

        ndvi = compute_ndvi(nir=ds.sel(band="B08").values, red=ds.sel(band="B04").values).astype(np.float32)

        s2_model = s2.with_data(s2.data.sel(band=DW_MODEL_BANDS))
        # Tag descriptions last: cloud_mask/ndvi derive from s2 via with_np, which
        # carries s2's metadata along — tagging s2 first would leak its description
        # into both, then clash with theirs.
        return {
            "sentinel_2_l1c": s2_model.with_metadata(
                {"description": f"Sentinel-2 L1C imagery ({len(DW_MODEL_BANDS)} bands, DynamicWorld input set)"}
            ),
            "cloud_mask": s2.with_np(mask).with_metadata(
                {"description": "Cloud and shadow mask, 0=clear, 1=cloud/shadow"}
            ),
            "ndvi": s2.with_np(ndvi).with_metadata(
                {"description": "Normalized Difference Vegetation Index"}
            ),
        }

    def context(self, tiles: dict[str, GeoTile]) -> dict[str, Any]:
        """PrithviTL's + Clay's raw forward() inputs, off this sample's sentinel tile.

        Sentinel tile = `tiles["sentinel_2_l1c"]` — the real STAC-sourced
        imagery layer. Datetime comes from `tile.times[0]` — the
        real per-scene acquisition timestamp off the loaded data's own
        `time` coordinate (`temporal_slots=1` on `sources` above guarantees
        exactly one) — not `tile.start`/`tile.end` (`GeoAnchor.datetime`),
        which `parse_datetime_range` widens a reduced-precision query date
        into a whole-day range; using that here would flatten every sample
        from this pipeline to midnight, losing the real acquisition hour
        `Clay`'s `time` wants and coarsening `PrithviTL`'s `temporal_coords`
        to day-of-year only.

        Every key/shape here mirrors a real forward() param name exactly —
        `temporal_coords`/`location_coords` for `PrithviTL.forward_pyramid`
        (mandatory ctx keys, see `encoder/prithvi.py`), `time`/`latlon`/`gsd`
        for `Clay.forward` (raw, normalized internally — see `encoder/clay.py`).
        `Clay.forward_pyramid` doesn't declare `time`/`latlon`/`gsd` as ctx
        requirements (its own docstring explains why), so those three just
        sit unused in ctx for a Clay chain — no model-specific reshape stage
        downstream either way, `ContextChain` merges this dict straight into
        whichever stage's own params match.

        Args:
            tiles: Layer name to GeoTile map — same sample `preprocess` built.

        Returns:
            Per-sample (no batch dim — `stack_samples` adds it):
                temporal_coords: (num_frames=1, 2) float32, (year,
                    day_of_year), day_of_year 0-indexed (Jan 1st = 0).
                location_coords: (2,) float32, (lat, lon) in degrees.
                time: (2,) float32, raw (iso_week, hour) — Clay normalizes internally.
                latlon: (2,) float32, raw (lat, lon) in degrees — Clay normalizes internally.
                gsd: scalar float32, this tile's real resolution in meters.
        """
        tile = tiles["sentinel_2_l1c"]
        lon, lat = tile.centroid
        acquired = tile.times[0]
        day_of_year = acquired.timetuple().tm_yday - 1  # tm_yday is 1-indexed; Prithvi wants 0-indexed
        return {
            "temporal_coords": torch.tensor([[acquired.year, day_of_year]], dtype=torch.float32),
            "location_coords": torch.tensor([lat, lon], dtype=torch.float32),
            "time": torch.tensor([acquired.isocalendar().week, acquired.hour], dtype=torch.float32),
            "latlon": torch.tensor([lat, lon], dtype=torch.float32),
        }
