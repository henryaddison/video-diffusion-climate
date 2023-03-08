import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, num_to_groups
import xarray as xr

MILESTONE = 195

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

index = 9999
x_orig = torch.from_numpy(xr.open_dataset("/user/work/cj19328/train_test/test.nc").pr.isel(time=range(index, index + 10)).values.reshape(1, 1, 10, 64, 64))
indices_a = [0, 3, 6, 9]
x_a = x_orig[:, :, indices_a]

if torch.cuda.is_available():
    diffusion = diffusion.cuda()
    x_a = x_a.cuda()

x_cond_sample=diffusion.sample_cond(x_a, indices_a)
torch.save(x_orig, f"/user/home/cj19328/cond_sample/original.pt")
torch.save(x_cond_sample, f"/user/home/cj19328/cond_sample/cond_sample.pt")

# if torch.cuda.is_available():
#     diffusion = diffusion.cuda()

# indices_a = [0, 1, 2, 3, 4]

# indices_b = [i for i in range(10) if i not in indices_a]
# offset = 5

# sample = diffusion.sample(batch_size = 1).cpu()
# while True:
#     torch.save(sample, f"/user/home/cj19328/cond_sample/long_video.pt")
#     x_a = sample[:, :, [i + offset for i in indices_a]].cuda()
#     x_cond_sample=diffusion.sample_cond(x_a, indices_a).cpu()
#     sample = torch.cat([sample, x_cond_sample[:, :, indices_b]], dim = 2)
#     offset += 5