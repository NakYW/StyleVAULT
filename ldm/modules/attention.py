from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def calc_token_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    mean = feat.mean(dim=2, keepdim=True)
    var = feat.var(dim=2, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    return mean, std


def token_mean_variance_norm(feat: torch.Tensor) -> torch.Tensor:
    mean, std = calc_token_mean_std(feat)
    return (feat - mean) / std


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.attn = None
        self.Q_c = None
        self.K_c = None
        self.V_c = None
        self.Q_s = None
        self.K_s = None
        self.V_s = None
        self.Q_cs = None
        self.K_cs = None
        self.V_cs = None
        self.Q_hat_cs = None
        self.ada_gate = nn.Parameter(torch.tensor(0.2))
        # toggle hooks for staged AdaAttN behaviour
        self.use_mean_shift = True      # Step 1: mean-only modulation
        self.use_std_scale =   True      # Step 2: add std-based affine scaling
        self.use_residual_gate = True   # Step 3: gated blend with base attention
        self.style_scale = 0.85         # Scaling factor for style values
        self.u_map = None
        self.pos_map = None
        self.bnd_map = None
        self.omega_map = None
        self.lambda_map = None
        self.eta_map = None
        self.dominance_map = None
        self.entropy_map = None
        self.margin_map = None
        self.structure_map = None
        self.attn_joint = None
        self.attn_style = None
        self.attn_content = None
        self.Q_final = None

    def _expand_injected_to_bh(self, tensor, batch_size, heads):
        if tensor is None:
            return None
        if tensor.dim() != 3:
            raise ValueError(f"Injected tensor must be 3D, got shape {tuple(tensor.shape)}")
        bh = batch_size * heads
        leading = tensor.shape[0]
        if leading == bh:
            return tensor
        if leading == heads:
            return repeat(tensor, 'h n d -> (b h) n d', b=batch_size)
        if leading == 1:
            return tensor.expand(bh, -1, -1)
        raise ValueError(
            f"Injected tensor leading dimension must be heads({heads}) or batch*heads({bh}), got {leading}"
        )

    def _safe_topk_mask(self, logits, k):
        # Keep only per-query top-k logits and hard-mask the rest.
        if logits.dim() < 2:
            raise ValueError(f"Logits must have at least 2 dims, got shape {tuple(logits.shape)}")
        ns = logits.shape[-1]
        k = int(k)
        if k >= ns:
            return logits
        if k <= 0:
            return torch.full_like(logits, torch.finfo(logits.dtype).min)
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        masked.scatter_(-1, topk_idx, topk_vals)
        return masked

    def _entropy_from_logits(self, logits):
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        return entropy

    def _top2_margin_from_logits(self, logits):
        last_dim = logits.shape[-1]
        if last_dim <= 1:
            top1 = torch.max(logits.float(), dim=-1, keepdim=True).values
            return top1
        top2 = torch.topk(logits.float(), k=2, dim=-1).values
        return top2[..., :1] - top2[..., 1:2]

    def _structure_score_from_query(self, q):
        qf = q.float()
        token_var = qf.var(dim=-1, unbiased=False, keepdim=True)
        token_norm = torch.sqrt(torch.clamp((qf * qf).mean(dim=-1, keepdim=True), min=1e-8))
        score = 0.5 * token_var + 0.5 * token_norm
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (score - mean) / std

    def _compute_uncertainty_map(self, style_logits, content_logits, q_content, heads, cfg):
        # Compute query-wise uncertainty from style/content competition signals.
        bh, n_q, _ = style_logits.shape
        if bh % heads != 0:
            raise ValueError(f"Leading dim {bh} is not divisible by heads={heads}")
        b = bh // heads

        style_logits_f = style_logits.float()
        content_logits_f = content_logits.float()
        dominance = (
            torch.logsumexp(style_logits_f, dim=-1, keepdim=True)
            - torch.logsumexp(content_logits_f, dim=-1, keepdim=True)
        )
        entropy = self._entropy_from_logits(style_logits_f)
        margin = self._top2_margin_from_logits(style_logits_f)
        if q_content is None:
            structure = torch.zeros_like(dominance)
        else:
            structure = self._structure_score_from_query(q_content)

        alpha_h = float(cfg.get('alpha_h', 1.0))
        alpha_d = float(cfg.get('alpha_d', 1.0))
        alpha_m = float(cfg.get('alpha_m', 1.0))
        alpha_e = float(cfg.get('alpha_e', 0.3))
        bias = float(cfg.get('uncertainty_bias', 0.0))

        u_raw = alpha_h * entropy - alpha_d * dominance - alpha_m * margin + alpha_e * structure + bias
        u = torch.sigmoid(u_raw).clamp(1e-4, 1.0 - 1e-4)

        def head_mean_map(x):
            x = rearrange(x, '(b h) n c -> b h n c', h=heads)
            return x.mean(dim=1, keepdim=True)

        u_map = head_mean_map(u)
        pos_map = 1.0 - u_map
        bnd_map = u_map
        dominance_map = head_mean_map(dominance)
        entropy_map = head_mean_map(entropy)
        margin_map = head_mean_map(margin)
        structure_map = head_mean_map(structure)

        return {
            'u_map': u_map,
            'pos_map': pos_map,
            'bnd_map': bnd_map,
            'dominance': dominance_map,
            'entropy': entropy_map,
            'margin': margin_map,
            'structure': structure_map,
        }

    def _reset_adaptive_debug(self):
        self.u_map = None
        self.pos_map = None
        self.bnd_map = None
        self.omega_map = None
        self.lambda_map = None
        self.eta_map = None
        self.dominance_map = None
        self.entropy_map = None
        self.margin_map = None
        self.structure_map = None
        self.attn_joint = None
        self.attn_style = None
        self.attn_content = None
        self.Q_final = None

    def forward(self,
                x,
                context=None,
                mask=None,
                Q_c_injected=None,
                K_s_injected=None,
                V_s_injected=None,
                injection_config=None,):
        self._reset_adaptive_debug()
        self.attn = None
        h = self.heads
        b = x.shape[0]
        cfg = dict(injection_config) if injection_config is not None else {}
        attn_matrix_scale = float(cfg.get('T', 1.0))
        q_mix = float(cfg.get('gamma', 0.0))
        omega_base = float(cfg.get('omega_base', q_mix if 'gamma' in cfg else 0.5))
        omega_base = max(0.0, min(1.0, omega_base))
        omega_min = float(cfg.get('omega_min', 0.05))
        omega_max = float(cfg.get('omega_max', 0.95))
        if omega_min > omega_max:
            omega_min, omega_max = omega_max, omega_min
        tau_s = float(cfg.get('tau_s', attn_matrix_scale if 'T' in cfg else 1.5))
        tau_s = max(tau_s, 1e-4)
        lambda_max = float(cfg.get('lambda_max', 1.8))
        lambda_max = max(1.0, lambda_max)
        eta_base = float(cfg.get('eta_base', 1.0))
        style_topk = int(cfg.get('style_topk', 128))
        style_scale = float(cfg.get('style_scale', self.style_scale))
        enable_adaptive_apem = bool(cfg.get('enable_adaptive_apem', True))

        context_in = context
        context = default(context, x)
        is_self_attention = (
            context_in is None
            or context is x
            or (
                context.shape == x.shape
                and context.device == x.device
                and context.data_ptr() == x.data_ptr()
            )
        )

        Q_c = None
        K_s = None
        V_s = None
        Q_cs = None
        K_cs = None
        V_cs = None

        Q_hat_cs = None
        Q_cs = self.to_q(x)
        Q_cs = rearrange(Q_cs, 'b n (h d) -> (b h) n d', h=h)
        K_cs = self.to_k(context)
        K_cs = rearrange(K_cs, 'b m (h d) -> (b h) m d', h=h)
        V_cs = self.to_v(context)
        V_cs = rearrange(V_cs, 'b m (h d) -> (b h) m d', h=h)

        Q_c_expanded = None
        K_s_expanded = None
        V_s_expanded = None
        if Q_c_injected is not None:
            Q_c_expanded = self._expand_injected_to_bh(Q_c_injected, b, h).to(
                device=Q_cs.device, dtype=Q_cs.dtype
            )
        if K_s_injected is not None:
            K_s_expanded = self._expand_injected_to_bh(K_s_injected, b, h).to(
                device=K_cs.device, dtype=K_cs.dtype
            )
        if V_s_injected is not None:
            V_s_expanded = self._expand_injected_to_bh(V_s_injected, b, h).to(
                device=V_cs.device, dtype=V_cs.dtype
            )

        use_adaptive_path = (
            is_self_attention
            and enable_adaptive_apem
            and Q_c_expanded is not None
            and K_s_expanded is not None
            and V_s_expanded is not None
        )

        if use_adaptive_path:
            # Adaptive APEM path: query-wise control + style/content joint softmax competition.
            Q_c = Q_c_expanded
            K_s = K_s_expanded
            V_s = V_s_expanded

            # Probe stage to estimate uncertainty maps before final query fusion.
            Q_probe = (1.0 - omega_base) * Q_c + omega_base * Q_cs
            S_probe = einsum('b i d, b j d -> b i j', Q_probe, K_s) * self.scale * tau_s
            C_probe = einsum('b i d, b j d -> b i j', Q_probe, K_cs) * self.scale

            uncertainty = self._compute_uncertainty_map(
                style_logits=S_probe,
                content_logits=C_probe,
                q_content=Q_cs,
                heads=h,
                cfg=cfg,
            )
            self.u_map = uncertainty['u_map']
            self.pos_map = uncertainty['pos_map']
            self.bnd_map = uncertainty['bnd_map']
            self.dominance_map = uncertainty['dominance']
            self.entropy_map = uncertainty['entropy']
            self.margin_map = uncertainty['margin']
            self.structure_map = uncertainty['structure']

            u_map = self.u_map.to(dtype=Q_cs.dtype, device=Q_cs.device)
            omega_map = omega_min + (omega_max - omega_min) * (1.0 - u_map)
            lambda_map = 1.0 + (lambda_max - 1.0) * (1.0 - u_map)
            eta_map = eta_base * (1.0 - u_map)

            self.omega_map = omega_map
            self.lambda_map = lambda_map
            self.eta_map = eta_map

            # Query-wise fused query for final attention competition.
            Q_c_head = rearrange(Q_c, '(b h) n d -> b h n d', h=h)
            Q_cs_head = rearrange(Q_cs, '(b h) n d -> b h n d', h=h)
            omega_head = omega_map.expand(-1, h, -1, -1).to(dtype=Q_cs.dtype, device=Q_cs.device)
            Q_final_head = (1.0 - omega_head) * Q_c_head + omega_head * Q_cs_head
            Q_final = rearrange(Q_final_head, 'b h n d -> (b h) n d')
            self.Q_final = Q_final

            # Joint competition: sparse style logits + dense content logits in one softmax space.
            S = einsum('b i d, b j d -> b i j', Q_final, K_s) * self.scale * tau_s
            C = einsum('b i d, b j d -> b i j', Q_final, K_cs) * self.scale

            if exists(mask):
                mask_ = rearrange(mask, 'b ... -> b (...)')
                mask_ = repeat(mask_, 'b j -> (b h) () j', h=h)
                C = C.masked_fill(~mask_, torch.finfo(C.dtype).min)

            lambda_bh = repeat(lambda_map, 'b 1 n 1 -> (b h) n 1', h=h).to(dtype=S.dtype, device=S.device)
            S_scaled = S * lambda_bh
            S_sparse = self._safe_topk_mask(S_scaled, style_topk)

            ns = S_sparse.shape[-1]
            L = torch.cat([S_sparse, C], dim=-1)
            A = torch.softmax(L.float(), dim=-1).to(dtype=L.dtype)
            A_style = A[..., :ns]
            A_content = A[..., ns:]

            V_style = V_s * style_scale
            V_joint = torch.cat([V_style, V_cs], dim=1)
            out = einsum('b i j, b j d -> b i d', A, V_joint)

            self.attn = A
            self.attn_joint = A
            self.attn_style = A_style
            self.attn_content = A_content
            Q_hat_cs = Q_final

            O_head = rearrange(out, '(b h) n d -> b h n d', h=h)
            if self.use_mean_shift or self.use_std_scale:
                A_head = rearrange(A, '(b h) n m -> b h n m', h=h).float()
                V_joint_head = rearrange(V_joint, '(b h) m d -> b h m d', h=h).float()
                mu = torch.einsum('bhnm,bhmd->bhnd', A_head, V_joint_head)
                Q_norm = token_mean_variance_norm(Q_final_head.float())
                out_mod_head = O_head
                if self.use_mean_shift:
                    out_mod_head = (Q_norm + mu).to(dtype=O_head.dtype, device=O_head.device)
                if self.use_std_scale:
                    nu = torch.einsum('bhnm,bhmd->bhnd', A_head, V_joint_head * V_joint_head)
                    sigma = torch.sqrt(torch.clamp(nu - mu * mu, min=1e-5))
                    out_mod_head = (sigma * Q_norm + mu).to(dtype=O_head.dtype, device=O_head.device)
            else:
                out_mod_head = O_head

            if self.use_residual_gate:
                global_gate = torch.tanh(self.ada_gate).to(dtype=O_head.dtype, device=O_head.device)
                local_gate = eta_map.expand(-1, h, -1, -1).to(dtype=O_head.dtype, device=O_head.device)
                gate = global_gate * local_gate
                O_head = O_head + gate * (out_mod_head - O_head)
            else:
                O_head = out_mod_head

            out = rearrange(O_head, 'b h n d -> (b h) n d')
        else:
            # Legacy path: keep original behavior when adaptive path is not enabled.
            if Q_c_expanded is None:
                Q_hat_cs = Q_cs
            else:
                Q_c = Q_c_expanded
                Q_hat_cs = Q_c * q_mix + Q_cs * (1.0 - q_mix)

            if K_s_expanded is None:
                k = K_cs
            else:
                K_s = K_s_expanded
                k = K_s

            if V_s_expanded is None:
                v = V_cs
            else:
                V_s = V_s_expanded
                v = V_s

            v = v * style_scale
            q = Q_hat_cs

            sim = einsum('b i d, b j d -> b i j', q, k)
            if Q_c_expanded is not None or K_s_expanded is not None:
                sim *= attn_matrix_scale
            sim *= self.scale

            if exists(mask):
                mask_ = rearrange(mask, 'b ... -> b (...)')
                mask_ = repeat(mask_, 'b j -> (b h) () j', h=h)
                sim = sim.masked_fill(~mask_, torch.finfo(sim.dtype).min)

            attn = sim.softmax(dim=-1)
            self.attn = attn
            out = einsum('b i j, b j d -> b i d', attn, v)

            Attn_cs_head = rearrange(attn, '(b h) n m -> b h n m', h=h)
            V_s_head = rearrange(v, '(b h) m d -> b h m d', h=h)
            Q_hat_cs_head = rearrange(q, '(b h) n d -> b h n d', h=h)

            V_cs_mean_head = torch.einsum('bhnm,bhmd->bhnd', Attn_cs_head, V_s_head)
            Q_hat_cs_norm_head = token_mean_variance_norm(Q_hat_cs_head)
            V_cs_mean_tokens = Q_hat_cs_norm_head + V_cs_mean_head
            V_cs_mean_tokens = rearrange(V_cs_mean_tokens, 'b h n d -> (b h) n d')

            out_mod = out
            if self.use_mean_shift:
                out_mod = V_cs_mean_tokens

            if self.use_std_scale:
                V_cs_sq_mean_head = torch.einsum('bhnm,bhmd->bhnd', Attn_cs_head, V_s_head * V_s_head)
                V_cs_std_head = torch.sqrt(torch.clamp(V_cs_sq_mean_head - V_cs_mean_head ** 2, min=1e-5))
                V_cs_affine_head = V_cs_std_head * Q_hat_cs_norm_head + V_cs_mean_head
                V_cs_affine_tokens = rearrange(V_cs_affine_head, 'b h n d -> (b h) n d')
                out_mod = V_cs_affine_tokens

            if self.use_residual_gate:
                gate = torch.tanh(self.ada_gate)
                out_mod = out + gate * (out_mod - out)

            out = out_mod

        self.Q_c = Q_c if Q_c is not None else Q_hat_cs
        self.K_c = K_cs
        self.V_c = V_cs
        self.Q_s = None
        self.K_s = K_s if K_s is not None else K_cs
        self.V_s = V_s if V_s is not None else V_cs
        self.Q_cs = Q_cs
        self.K_cs = K_cs
        self.V_cs = V_cs
        self.Q_hat_cs = Q_hat_cs

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self,
                x,
                context=None,
                self_attn_Q_c_injected=None,
                self_attn_K_s_injected=None,
                self_attn_V_s_injected=None,
                injection_config=None,
                ):
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_Q_c_injected,
                                          self_attn_K_s_injected,
                                          self_attn_V_s_injected,
                                          injection_config,), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_Q_c_injected=None,
                 self_attn_K_s_injected=None,
                 self_attn_V_s_injected=None,
                 injection_config=None):
        x_ = self.attn1(self.norm1(x),
                       Q_c_injected=self_attn_Q_c_injected,
                       K_s_injected=self_attn_K_s_injected,
                       V_s_injected=self_attn_V_s_injected,
                       injection_config=injection_config,)
        x = x_ + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_Q_c_injected=None,
                self_attn_K_s_injected=None,
                self_attn_V_s_injected=None,
                injection_config=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_Q_c_injected=self_attn_Q_c_injected,
                      self_attn_K_s_injected=self_attn_K_s_injected,
                      self_attn_V_s_injected=self_attn_V_s_injected,
                      injection_config=injection_config)

            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
