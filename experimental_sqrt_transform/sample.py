import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, num_to_groups

NUM_SAMPLES = 16384
BATCH_SIZE = 16
MILESTONE = 200

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

checkpoint = torch.load(f"/user/work/cj19328/results_sqrt_lambda_30/model-{MILESTONE}.pt")
diffusion.load_state_dict(checkpoint['ema'])

samples=torch.empty(0, 1, 10, 64, 64)

if torch.cuda.is_available():
    diffusion = diffusion.cuda()
    samples = samples.cuda()

batch_sizes=num_to_groups(NUM_SAMPLES, BATCH_SIZE)
for batch_size in batch_sizes:
    new_samples=diffusion.sample(batch_size = batch_size)
    samples=torch.cat((samples, new_samples), dim = 0)
    torch.save(samples, f"/user/home/cj19328/samples_sqrt.pt")