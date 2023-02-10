import os
import sys
import numpy as np
import torch
import xarray as xr

FRAMES_PER_TENSOR = 10

if __name__ == "__main__":
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]

    src_path = os.path.join(src_dir, "*.nc")
    pr = xr.open_mfdataset(src_path).pr.values

    pr = np.sqrt(pr)

    # batch channels frames height width
    new_shape = (
        pr.shape[1] // FRAMES_PER_TENSOR,
        1,
        FRAMES_PER_TENSOR,
        pr.shape[2],
        pr.shape[3]
    )
    
    pr_split = pr.reshape(new_shape)

    pr_split = torch.from_numpy(pr_split)
    torch.save(pr_split, os.path.join(dst_dir, "train.pt"))
