import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, num_to_groups
import xarray as xr

MILESTONE = 174

model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    num_frames=10,
    timesteps=1000,   # number of steps
)

checkpoint = torch.load(f"/user/work/cj19328/results/model-{MILESTONE}.pt")
diffusion.load_state_dict(checkpoint['ema'])

index = 5829
x_orig = torch.from_numpy(xr.open_dataset("/user/work/cj19328/train_test/test.nc").pr.isel(time=range(index, index + 10)).values.reshape(1, 1, 10, 64, 64))
mask_a = torch.tensor([True, False, False, True, False, False, True, False, False, True]).reshape(1, 1, 10, 1, 1)
x_a = x_orig * mask_a

if torch.cuda.is_available():
    diffusion = diffusion.cuda()
    x_a = x_a.cuda()
    mask_a = mask_a.cuda()

x_cond_sample=diffusion.sample_cond_replacement(x_a, mask_a)
torch.save(x_orig, f"/user/home/cj19328/cond_sample/original.pt")
torch.save(x_cond_sample, f"/user/home/cj19328/cond_sample/cond_sample.pt")