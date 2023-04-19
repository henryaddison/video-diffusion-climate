import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from einops import rearrange
from einops_exts import rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from matplotlib import pyplot as plt

PR_MAX = 76.96769714355469
# PR_MAX = 8.773123741149902
PR_MIN = 0.

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        attn_heads = 8,
        attn_dim_head = 32,
        init_dim = None,
        init_kernel_size = 7,
        resnet_groups = 8
    ):
        super().__init__()

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))

        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert init_kernel_size % 2 == 1

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(1, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, 1, 1)
        )

    def forward(
        self,
        x,
        time,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)






class MonotonicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Tanh()

        self.layer_1 = nn.Linear(1, 16)
        self.layer_2 = nn.Linear(16, 16)
        self.layer_3 = nn.Linear(16, 1)

        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.xavier_uniform_(self.layer_3.weight)

        self.enforce_monotonicity()

    def forward(self, x):
        identity = x

        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)
        x = self.act(x)
        x = self.layer_3(x)
        x = self.act(x)
        x = x + 0.001 * identity

        return x
    
    def normalise(self, y):
        y_max = self(torch.tensor([2.]).cuda()).item()
        y_min = self(torch.tensor([0.]).cuda()).item()
        return 2 * ((y - y_min) / (y_max - y_min)) - 1

    def enforce_monotonicity(self):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                param.data.clamp_(min=0)

    def log_dy_dx(self, x, y):
        dy_dx = torch.autograd.grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
        log_dy_dx = torch.log(dy_dx)
        # log_dy_dx *= x > 2 * ((0.01 - PR_MIN) / (PR_MAX - PR_MIN))
        return log_dy_dx.mean()

    @torch.inference_mode()
    def plot(self):
        xs = torch.linspace(PR_MIN, PR_MAX, 100000).reshape(-1, 1).cuda()
        xs_normalised = 2 * ((xs - PR_MIN) / (PR_MAX - PR_MIN))
        ys = self.forward(xs_normalised)
        print("ys_min, ys_max:", ys[0].item(), ys[-1].item())
        ys = self.normalise(ys)
        torch.save(torch.stack([xs.flatten(), ys.flatten()], dim=1).cpu(), "transform_0.1_lambda_30.pt")
        plt.figure()
        plt.plot(xs.cpu(), ys.cpu())
        plt.savefig("monotonic_net_transform_0.1_lambda_30.png")
        plt.close()


# gaussian diffusion trainer class

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        unet,
        *,
        image_size,
        num_frames,
        num_timesteps = 1000
    ):
        super().__init__()
        self.unet = unet
        self.image_size = image_size
        self.num_frames = num_frames
        self.num_timesteps = num_timesteps
        self.monotonic_net = MonotonicNet()
        self.omega_r = 100000 # Reconstruction-guided sampling

    def log_snr_schedule_cosine(self, t, log_snr_min = -30, log_snr_max = 30):
        b = t.shape[0]
        t_min = math.atan(math.exp(-0.5 * log_snr_max))
        t_max = math.atan(math.exp(-0.5 * log_snr_min))
        log_snr = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
        log_snr = log_snr.reshape(b, 1, 1, 1, 1)
        return log_snr

    def log_snr_to_alpha_sigma(self, log_snr):
        alpha = torch.sqrt(torch.sigmoid(log_snr))
        sigma = torch.sqrt(torch.sigmoid(-log_snr))
        return alpha, sigma

    def predict_x_hat(self, z_t, v_hat_t, lambda_t):
        alpha_t, sigma_t = self.log_snr_to_alpha_sigma(lambda_t)
        return alpha_t * z_t - sigma_t * v_hat_t

    def q_posterior(self, z_t, x, lambda_s, lambda_t):
        alpha_s, sigma_s = self.log_snr_to_alpha_sigma(lambda_s)
        alpha_t, _ = self.log_snr_to_alpha_sigma(lambda_t)

        # alpha_ts = alpha_t / alpha_s
        # sigma_ts = torch.sqrt(sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2)

        # mu_st = ((alpha_ts * sigma_s ** 2) / sigma_t ** 2) * z_t + ((alpha_s * sigma_ts ** 2) / sigma_t ** 2) * x
        # sigma_st = (sigma_ts * sigma_s) / sigma_t
        mu_st = torch.exp(lambda_t - lambda_s) * (alpha_s / alpha_t) * z_t + (-torch.expm1(lambda_t - lambda_s)) * alpha_s * x
        sigma_st = torch.sqrt((-torch.expm1(lambda_t - lambda_s)) * sigma_s ** 2)
        
        return mu_st, sigma_st

    def p_mean_variance(self, z_t, lambda_s, lambda_t, x_a = None, indices_a = None):
        if x_a is not None: # Reconstruction-guided sampling
            indices_b = [i for i in range(0, self.num_frames) if i not in indices_a]
            z_t = z_t.detach()
            z_t_b = z_t[:, :, indices_b]
            z_t_b.requires_grad = True
            z_t[:, :, indices_b] = z_t_b

        v_hat_t = self.unet(z_t, lambda_t.reshape(-1))
        x_hat = self.predict_x_hat(z_t, v_hat_t, lambda_t)

        if x_a is not None: # Reconstruction-guided sampling
            alpha_t, _ = self.log_snr_to_alpha_sigma(lambda_t)
            x_hat_a = x_hat[:, :, indices_a]
            error = F.mse_loss(x_a, x_hat_a)
            grad = torch.autograd.grad(outputs = error, inputs = z_t_b)[0]
            x_hat[:, :, indices_b] -= ((self.omega_r * alpha_t) / 2) * grad

        # x_hat = x_hat.clamp(-1, 1)
        mu_st, sigma_st = self.q_posterior(z_t, x_hat, lambda_s, lambda_t)
        return mu_st, sigma_st

    def p_sample(self, z_t, lambda_s, lambda_t, x_a = None, indices_a = None):
        mu_st, sigma_st = self.p_mean_variance(z_t, lambda_s, lambda_t, x_a = x_a, indices_a = indices_a)
        noise = torch.randn_like(z_t)
        
        return mu_st + sigma_st * noise

    def p_sample_loop(self, shape):
        z_t = torch.randn(shape).cuda()
        b = shape[0]

        for i in tqdm(reversed(range(1, self.num_timesteps + 1)), desc='sampling loop time step', total=self.num_timesteps):
            s = torch.full((b,), (i - 1) / self.num_timesteps).cuda()
            t = torch.full((b,), i / self.num_timesteps).cuda()
            lambda_s = self.log_snr_schedule_cosine(s)
            lambda_t = self.log_snr_schedule_cosine(t)
            z_t = self.p_sample(z_t, lambda_s, lambda_t)

        x = z_t.clamp_(-1, 1)

        return x

    @torch.inference_mode()
    def sample(self, batch_size = 16):
        self.monotonic_net.plot()

        samples = self.p_sample_loop((batch_size, 1, self.num_frames, self.image_size, self.image_size))
        return samples

    def sample_recon_guidance(self, x_a, indices_a):
        b = x_a.shape[0]
        z_t = torch.randn((b, 1, self.num_frames, self.image_size, self.image_size)).cuda()

        for i in tqdm(reversed(range(1, self.num_timesteps + 1)), desc='sampling loop time step', total=self.num_timesteps):
            s = torch.full((b,), (i - 1) / self.num_timesteps).cuda()
            t = torch.full((b,), i / self.num_timesteps).cuda()
            lambda_s = self.log_snr_schedule_cosine(s)
            lambda_t = self.log_snr_schedule_cosine(t)
            z_t = self.p_sample(z_t, lambda_s, lambda_t, is_final_step = i == 1, x_a = x_a, indices_a = indices_a)

            z_t[:, :, indices_a] = self.q_sample_recon_guidance(z_t[:, :, indices_a], x_a, lambda_s, lambda_t)

        z_t[:, :, indices_a] = x_a

        return z_t

    def q_sample_recon_guidance(self, z_t_a, x_a, lambda_s, lambda_t):
        noise = torch.randn_like(z_t_a)
        mu_st, sigma_st = self.q_posterior(z_t_a, x_a, lambda_s, lambda_t)
        return mu_st + sigma_st * noise

    def q_sample(self, x, lambda_t, noise):
        alpha_t, sigma_t = self.log_snr_to_alpha_sigma(lambda_t)
        return alpha_t * x + sigma_t * noise
        
    def p_losses(self, x):
        b = x.shape[0]

        x.requires_grad = True
        x_shape = x.shape
        y = self.monotonic_net(x.reshape(*x_shape, 1))
        y = y.reshape(*x_shape)
        y = self.monotonic_net.normalise(y)

        log_dy_dx = self.monotonic_net.log_dy_dx(x, y)

        times = torch.zeros(b).uniform_(0, 1).cuda() # TODO: Maybe implement the approach outlined in VDM paper
        lambda_t = self.log_snr_schedule_cosine(times)

        noise = torch.randn_like(x)
        y_noisy = self.q_sample(y, lambda_t, noise)
        v = self.unet(y_noisy, lambda_t.reshape(-1))

        alpha_t, sigma_t = self.log_snr_to_alpha_sigma(lambda_t)
        v_target = alpha_t * noise - sigma_t * y

        diffusion_loss = F.mse_loss(v, v_target)
        loss = diffusion_loss - log_dy_dx
        print("Diffusion loss and -log|dy/dx|", diffusion_loss.item(), -log_dy_dx.item())

        return loss

    def forward(self, x):
        x = 2 * ((x - PR_MIN) / (PR_MAX - PR_MIN)) # In the range [0, 2] just for better interpretability of log|dy / dx|

        return self.p_losses(x)


class Dataset(data.Dataset):
    def __init__(
        self,
        train_path,
    ):
        super().__init__()
        self.tensor = torch.load(train_path)

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        return self.tensor[index]

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_path,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = '/results',
        num_samples = 16,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = Dataset(train_path)

        print(f'found {len(self.ds)} videos as tensors in {train_path}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.num_samples = num_samples
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
    ):
        while self.step < self.train_num_steps:
            for _ in range(self.gradient_accumulate_every):
                data = next(self.dl)
                if torch.cuda.is_available():
                    data = data.cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(data)

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
            self.model.monotonic_net.enforce_monotonicity()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(self.num_samples, self.batch_size)

                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)
                
                video_path = str(self.results_folder / str(f'{milestone}.pt'))
                torch.save(all_videos_list, video_path)
                
                self.save(milestone)

            self.step += 1

        print('training completed')