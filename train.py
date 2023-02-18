import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

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

if torch.cuda.is_available():
    diffusion = diffusion.cuda()

trainer = Trainer(
    diffusion,
    # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    '/user/work/cj19328/train_2r.pt',
    train_batch_size=32,
    train_lr=1e-4,
    save_and_sample_every=1000,
    train_num_steps=700000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True,                       # turn on mixed precision
    results_folder='/user/work/cj19328/results_test'        # folder to save results
)

# trainer.load(28)
trainer.train()
