import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, num_to_groups

NUM_SAMPLES = 16384
BATCH_SIZE = 16
MILESTONE = 218

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

checkpoint = torch.load(f"/user/work/cj19328/results_transform_0.1_lambda_30/model-{MILESTONE}.pt")
diffusion.load_state_dict(checkpoint['ema'])

samples=torch.empty(0, 1, 10, 64, 64)

if torch.cuda.is_available():
    diffusion = diffusion.cuda()
    samples = samples.cuda()
    

# TODO: Remove this, and implement in the model if it produces good samples
# It's squashed together to deliberately look ugly so hopefully I remember to implement this properly
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


batch_sizes=num_to_groups(NUM_SAMPLES, BATCH_SIZE)
for batch_size in batch_sizes:
    new_samples=diffusion.sample(batch_size = batch_size)
    new_samples = inverse_tensor(new_samples).cuda()
    samples=torch.cat((samples, new_samples), dim = 0)
    torch.save(samples, f"/user/home/cj19328/samples.pt")