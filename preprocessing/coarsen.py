import os
import sys
import math
import numpy as np
import xarray as xr
import cartopy.crs as ccrs


class SelectDomain:

    # Domain centres in standard latitude and longitude
    DOMAIN_CENTRES_LON_LAT = {
        "london": (-0.118092, 51.509865),
        "birmingham": (-1.898575, 52.489471),
    }

    # Domain centres in rotated pole latitude and longitude
    DOMAIN_CENTRES_RP_LONG_LAT = {
        domain_name: ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5).transform_point(
            *lon_lat, src_crs=ccrs.PlateCarree())
        for domain_name, lon_lat in DOMAIN_CENTRES_LON_LAT.items()
    }

    def __init__(self, subdomain, size=64):
        self.subdomain = subdomain
        self.size = size

    def run(self, ds):

        # Find the grid square in the dataset nearest to the centre
        # Assumes dataset is in rotated pole coordinates
        centre_rp_lon_lat = self.DOMAIN_CENTRES_RP_LONG_LAT[self.subdomain]
        centre_ds = ds.sel(
            grid_longitude=360.0 + centre_rp_lon_lat[0],
            grid_latitude=centre_rp_lon_lat[1],
            method="nearest",
        )

        # Find the INDEX of the central square in the grid latitude and longitude coordinates
        centre_long_idx = np.where(
            ds.grid_longitude.values == centre_ds.grid_longitude.values
        )[0].item()
        centre_lat_idx = np.where(
            ds.grid_latitude.values == centre_ds.grid_latitude.values
        )[0].item()

        # Set how many grid squares to go up, down, left and right from the centre box
        # in order to get a box the is self.size by self.size
        radius = self.size - 1
        left_length = math.floor(radius / 2.0)
        right_length = math.ceil(radius / 2.0)
        down_length = math.floor(radius / 2.0)
        up_length = math.ceil(radius / 2.0)

        # Select only the bits of the dataset that lie in the box defined
        ds = ds.sel(
            grid_longitude=slice(
                ds.grid_longitude[centre_long_idx - left_length].values,
                ds.grid_longitude[centre_long_idx + right_length].values,
            ),
            grid_latitude=slice(
                ds.grid_latitude[centre_lat_idx - down_length].values,
                ds.grid_latitude[centre_lat_idx + up_length].values,
            ),
        )

        return ds


def get_paths(src_dir, dst_dir, file):
    new_file = file.replace(
        "uk_2.2km",
        "birmingham-64_2.2km-coarsened-4x-2.2km-coarsened-4x"
    )
    src_path = os.path.join(src_dir, file)
    dst_path = os.path.join(dst_dir, new_file)
    return src_path, dst_path


def coarsen_and_save(src_path, dst_path):
    ds = xr.open_dataset(src_path)
    ds = ds.coarsen(grid_latitude=4, grid_longitude=4, boundary="trim").mean()
    ds = SelectDomain("birmingham", size=64).run(ds)
    ds.to_netcdf(dst_path)


if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    for file in os.listdir(src_dir):
        if file.endswith('.nc'):
            src_path, dst_path = get_paths(src_dir, dst_dir, file)
            coarsen_and_save(src_path, dst_path)
