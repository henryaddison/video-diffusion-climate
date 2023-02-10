import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 10,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

if torch.cuda.is_available():
    diffusion = diffusion.cuda()

trainer = Trainer(
    diffusion,
    './data',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 2,
    train_lr = 1e-4,
    save_and_sample_every = 500,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()