import os
import sys
import numpy as np
import torch
import xarray as xr

HIY = 12 * 30 * 24  # Hours in a year
FRAMES_PER_VIDEO = 10
IMAGE_DIM = 64


def create_video_chunks(arr):
    n_videos = HIY - FRAMES_PER_VIDEO + 1
    new_shape = (n_videos, FRAMES_PER_VIDEO, IMAGE_DIM, IMAGE_DIM)
    new_strides = (arr.strides[0], arr.strides[0], *arr.strides[1:])
    new_chunks = np.lib.stride_tricks.as_strided(
        arr,
        shape=new_shape,
        strides=new_strides
    )
    return new_chunks


if __name__ == "__main__":
    src_path = sys.argv[1]

    pr = xr.open_dataset(src_path).pr.values
    pr = pr.reshape(-1, HIY, IMAGE_DIM, IMAGE_DIM)

    chunks = np.empty(
        (0, FRAMES_PER_VIDEO, IMAGE_DIM, IMAGE_DIM),
        dtype=np.float32
    )
    for pr_year in pr:
        new_chunks = create_video_chunks(pr_year)
        chunks = np.concatenate((chunks, new_chunks), axis=0)

    chunks = chunks.reshape(-1, 1, FRAMES_PER_VIDEO, IMAGE_DIM, IMAGE_DIM)
    chunks = torch.from_numpy(chunks)

    filename = os.path.splitext(
        src_path)[0] + "_sliding-{FRAMES_PER_VIDEO}f-1s.pt"
    torch.save(chunks, filename)
