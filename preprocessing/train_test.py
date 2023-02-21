import os
import sys
import xarray as xr

HIS = 3 * 30 * 24  #  Hours in a season
TEST_RANGES = (range(i * HIS, (i + 1) * HIS) for i in range(0, 240, 5))
TRAIN_RANGES = (range(i * HIS, (i + 4) * HIS) for i in range(1, 240, 5))


def create_datasets(ds, dst_dir):
    test_ds_list = [ds.isel(time=test_range) for test_range in TEST_RANGES]
    train_ds_list = [ds.isel(time=train_range) for train_range in TRAIN_RANGES]

    test_ds = xr.concat(test_ds_list, dim="time")
    train_ds = xr.concat(train_ds_list, dim="time")

    test_ds.to_netcdf(os.path.join(dst_dir, "test.nc"))
    train_ds.to_netcdf(os.path.join(dst_dir, "train.nc"))


if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    src_path = os.path.join(src_dir, "*.nc")
    ds = xr.open_mfdataset(src_path)

    create_datasets(ds, dst_dir)
