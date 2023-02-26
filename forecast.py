import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, num_to_groups
import xarray as xr

MILESTONE = 122

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

index = 3333
x_orig = torch.from_numpy(xr.open_dataset("/user/work/cj19328/train_test/test.nc").pr.isel(time=range(index, index + 10)).values.reshape(1, 1, 10, 64, 64))
x_a = x_orig[0][0][:5]

if torch.cuda.is_available():
    diffusion = diffusion.cuda()
    x_a = x_a.cuda()

x_forecast=diffusion.sample_cond_replacement(x_a)
torch.save(x_orig, f"/user/home/cj19328/forecasts/original.pt")
torch.save(x_forecast, f"/user/home/cj19328/forecasts/forecast.pt")