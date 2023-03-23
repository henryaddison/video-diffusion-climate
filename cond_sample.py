import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, num_to_groups
import xarray as xr

MILESTONE = 139

model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    num_frames=10,
    num_timesteps=1000,   # number of steps
)

checkpoint = torch.load(f"/user/work/cj19328/results_transform_0.1/model-{MILESTONE}.pt")
diffusion.load_state_dict(checkpoint['ema'])

if torch.cuda.is_available():
    diffusion = diffusion.cuda()

# # TODO: Remove this, and implement in the model if it produces good samples
# # It's squashed together to deliberately look ugly so hopefully I remember to implement this properly
from sortedcontainers import SortedDict
PR_MAX = 76.96769714355469
PR_MIN = 0.
xs = torch.linspace(PR_MIN, PR_MAX, 100000).reshape(-1, 1).cuda()
xs_normalised = (2 * ((xs - PR_MIN) / (PR_MAX - PR_MIN)))
ys = diffusion.monotonic_net(xs_normalised)
ys = diffusion.monotonic_net.normalise(ys)
lookup_table = torch.stack([xs.flatten(), ys.flatten()], dim=1).tolist()
sorted_dict = SortedDict()
for x, y in lookup_table:
    sorted_dict[y] = x
def inverse_tensor(y_tensor):
    y_tensor_flat = y_tensor.flatten()
    x_values = []
    for y_target in y_tensor_flat:
        index = sorted_dict.bisect(y_target.item())
        if index == len(sorted_dict):
            index -= 1
        _, x_target = sorted_dict.peekitem(index)
        x_values.append(x_target)
    x_tensor = torch.tensor(x_values).reshape(y_tensor.shape)
    return x_tensor



# index = 5555
# x_orig = torch.from_numpy(xr.open_dataset("/user/work/cj19328/train_test/test.nc").pr.isel(time=range(index, index + 10)).values.reshape(1, 1, 10, 64, 64))
# indices_a = [0, 3, 6, 9]
# x_a = x_orig[:, :, indices_a]

# if torch.cuda.is_available():
#     diffusion = diffusion.cuda()
#     x_a = x_a.cuda()

# x_a_shape = x_a.shape
# x_a = 2 * ((x_a - PR_MIN) / (PR_MAX - PR_MIN))
# x_a = diffusion.monotonic_net(x_a.reshape(*x_a_shape, 1))
# x_a = x_a.reshape(*x_a_shape)
# x_a = diffusion.monotonic_net.normalise(x_a)
# print(x_a.min(), x_a.max())

# x_cond_sample=diffusion.sample_recon_guidance(x_a, indices_a)
# x_cond_sample=inverse_tensor(x_cond_sample)
# torch.save(x_orig, f"/user/home/cj19328/cond_sample/original.pt")
# torch.save(x_cond_sample, f"/user/home/cj19328/cond_sample/cond_sample.pt")




if torch.cuda.is_available():
    diffusion = diffusion.cuda()

indices_a = [0, 1, 2, 3, 4]
indices_b = [5, 6, 7, 8, 9]
offset = 5

sample = diffusion.sample(batch_size = 16).cpu()
print(sample.shape)
for _ in range(18):
    x_a = sample[:, :, [i + offset for i in indices_a]].cuda()
    x_cond_sample=diffusion.sample_recon_guidance(x_a, indices_a).cpu()
    sample = torch.cat([sample, x_cond_sample[:, :, indices_b]], dim = 2)
    print(sample.shape)
    offset += 5

sample = inverse_tensor(sample).reshape(*sample.shape)
torch.save(sample, f"/user/home/cj19328/cond_sample/long_video.pt")
print("Done!")