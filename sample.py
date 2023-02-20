import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, num_to_groups

NUM_SAMPLES = 1024
BATCH_SIZE = 4
MILESTONE = 75

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

milestone = 48
checkpoint = torch.load(f"/user/work/cj19328/results_test/model-{milestone}.pt"))
model.load_state_dict(checkpoint['ema'])

samples=torch.empty(0, 1, 64, 64)
batch_sizes=num_to_groups(NUM_SAMPLES, BATCH_SIZE)
for batch_size in batch_sizes:
    new_samples=model.sample(batch_size = batch_size)
    samples=torch.cat((samples, new_samples), dim = 0)
    torch.save(samples, f"/user/home/cj19328/samples.pt")
