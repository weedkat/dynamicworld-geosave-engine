from __future__ import annotations

from functools import cached_property

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

    This pipeline is a concrete example of how to use GeoPipeline to build a
    multi-layer sample from a single anchor. It fetches Sentinel-2 L1C imagery
    from the Copernicus Data Space Ecosystem STAC endpoint, derives a cloud/shadow mask 
    and NDVI from it, and returns a dict of GeoTile layers.
    """
    @cached_property
    def sources(self) -> dict[str, StacSource]:
        stac_client = StacClient.cdse()
        # temporal_slots=1 because dynamicworld dataset is per scene
        return {
            "sentinel_2_l1c": stac_client.source(
                "sentinel-2-l1c", bands=L1C_BANDS, max_nodata_fraction=0.1, temporal_slots=1
            )
        }

    def preprocess(self, raw: dict[str, GeoTile]) -> dict[str, GeoTile]:
        s2 = raw["sentinel_2_l1c"]
        # temporal_slots=1 on the source (see `sources` above) — exactly one
        # scene per sample, so drop straight to (band, y, x)
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

        s2_tile = (
            s2.with_data(s2.data.sel(band=DW_MODEL_BANDS))
            .with_metadata(
                {"description": f"Sentinel-2 L1C imagery ({len(DW_MODEL_BANDS)} bands, DynamicWorld input set)"}
            )
            .with_plot_meta(
                rgb_bands=("B04", "B03", "B02")
            )
        )

        cloud_mask_tile = (
            s2.with_np(mask).with_metadata(
                {"description": "Cloud and shadow mask"}
            )
            .with_plot_meta(
                class_map={0: "clear", 1: "cloud/shadow"}, color_map={0: "#FFFFFF", 1: "#000000"}
            )
        )

        ndvi_tile = s2.with_np(ndvi).with_metadata(
            {"description": "Normalized Difference Vegetation Index"}
        )

        return {
            "sentinel_2_l1c": s2_tile,
            "cloud_mask": cloud_mask_tile,
            "ndvi": ndvi_tile,
        }

    def context(self, tiles: dict[str, GeoTile]) -> dict[str, torch.Tensor]:
        """PrithviTL's + Clay's raw forward() inputs for this sample.

        Keys mirror real forward() param names: `temporal_coords`/
        `location_coords` for `PrithviTL.forward_pyramid`, `time`/
        `latlon` for `Clay.forward`. Clay doesn't require these as ctx
        keys, so they sit unused in a Clay chain — harmless, `ContextChain`
        only pulls what a stage's own forward declares.

        Args:
            tiles: Layer name to GeoTile map — same sample `preprocess` built.

        Returns:
            {
                "temporal_coords": (1, 2) float32, (year, day_of_year), 0-indexed,
                "location_coords": (2,) float32, (lat, lon) degrees,
                "time": (2,) float32, raw (iso_week, hour),
                "latlon": (2,) float32, raw (lat, lon) degrees,
            }
        """
        tile = tiles["sentinel_2_l1c"]
        lon, lat = tile.centroid
        acquired = tile.times[0]
        day_of_year = acquired.timetuple().tm_yday - 1  # tm_yday is 1-indexed; Prithvi wants 0-indexed
        return {
            "temporal_coords": torch.tensor([[acquired.year, day_of_year]], dtype=torch.float32), # (1, 2)
            "location_coords": torch.tensor([lat, lon], dtype=torch.float32), # (2,)
            "time": torch.tensor([acquired.isocalendar().week, acquired.hour], dtype=torch.float32), # (2,)
            "latlon": torch.tensor([lat, lon], dtype=torch.float32), # (2,)
        }
