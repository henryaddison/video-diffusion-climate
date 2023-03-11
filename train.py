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
    num_timesteps=1000,
)

if torch.cuda.is_available():
    diffusion = diffusion.cuda()

trainer = Trainer(
    diffusion,
    # '/user/work/cj19328/train_test/train_sliding-10f-1s_sqrt.pt',
    '/user/work/cj19328/train_test/train_sliding-10f-1s.pt',
    train_batch_size=32,
    train_lr=1e-4,
    save_and_sample_every=5000,
    train_num_steps=700000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True,
    results_folder='/user/work/cj19328/results_continuous_cosine_30'
)

# trainer.load(4)
trainer.train()
