"""MLX port of the SiT forward pass (sit/models.py), loading the same torch state_dict.

MLX is Apple's array framework for Apple-silicon GPUs. This file re-implements SiT's forward pass with plain MLX operations so
the speed benchmark can compare it with PyTorch on the same weights. It is not used by any guidance_* script.
Each step below mirrors a module in sit/models.py: PatchEmbed, TimestepEmbedder, LabelEmbedder, SiTBlock, FinalLayer, unpatchify.
"""
import math
import mlx.core as mx

CONFIGS = {"XL": (28, 1152, 16), "L": (24, 1024, 16), "B": (12, 768, 12), "S": (12, 384, 6)}  # size -> (depth, width, heads)


def _ln(x):
    """LayerNorm without learned scale or bias (SiT's norms use elementwise_affine=False), eps 1e-6 as in SiT."""
    return mx.fast.layer_norm(x, None, None, 1e-6)


def _lin(x, p, name):
    """nn.Linear using the torch weights: torch stores weight as (out, in), so it is transposed."""
    return x @ p[name + ".weight"].T + p[name + ".bias"]


class SiTMLX:
    def __init__(self, sd, name, input_size=32, in_channels=4, learn_sigma=True, dtype=mx.float32):
        size, patch = name.split("-")[1].split("/")  # "SiT-XL/2" -> ("XL", "2")
        self.depth, self.dim, self.heads = CONFIGS[size]
        self.patch, self.in_ch, self.learn_sigma = int(patch), in_channels, learn_sigma
        self.out_ch = in_channels * 2 if learn_sigma else in_channels  # 4 velocity channels + 4 learned-sigma channels
        self.grid = input_size // self.patch  # patches per side: 32 / 2 = 16
        self.dtype = dtype
        # Accepts either torch tensors or MLX arrays (worker.py passes mx.load output), cast to the requested precision.
        self.p = {k: (v if isinstance(v, mx.array) else mx.array(v.float().numpy())).astype(dtype)
                  for k, v in sd.items()}
        # The patch embedding is a Conv2d with kernel = stride = patch size, which is the same as a linear layer applied to each
        # flattened patch. Reshape its (D, C, P, P) kernel to (D, C*P*P) so it can be used with _lin.
        w = self.p["x_embedder.proj.weight"]
        self.p["x_embedder.proj.weight"] = w.reshape(w.shape[0], -1)
        mx.eval(list(self.p.values()))  # MLX is lazy; force the conversions now so they are not timed as part of sampling

    def t_embed(self, t):
        """TimestepEmbedder: a 256-d sinusoidal encoding of t (cos half, then sin half), then Linear -> SiLU -> Linear."""
        half = 128
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)  # geometric frequencies, as in transformers
        a = t[:, None].astype(mx.float32) * freqs[None]
        e = mx.concatenate([mx.cos(a), mx.sin(a)], axis=-1).astype(self.dtype)
        h = _lin(e, self.p, "t_embedder.mlp.0")
        return _lin(h * mx.sigmoid(h), self.p, "t_embedder.mlp.2")  # x * sigmoid(x) is SiLU

    def __call__(self, x, t, y):
        p, P, g = self.p, self.patch, self.grid
        N = x.shape[0]
        # Patchify: (N, C, 32, 32) -> (N, g, g, C, P, P) -> (N, g*g tokens, C*P*P). This flatten order matches the conv kernel above.
        x = x.astype(self.dtype).reshape(N, self.in_ch, g, P, g, P)
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(N, g * g, -1)
        x = _lin(x, p, "x_embedder.proj") + p["pos_embed"]
        # Conditioning vector: time embedding plus class embedding (a plain table lookup, which is all LabelEmbedder does in eval mode).
        c = self.t_embed(t) + p["y_embedder.embedding_table.weight"][y]
        sc = c * mx.sigmoid(c)  # SiLU(c): every adaLN_modulation starts with SiLU, so compute it once
        H, D = self.heads, self.dim // self.heads
        for i in range(self.depth):
            b = f"blocks.{i}."
            # adaLN-Zero: c produces a shift, scale and gate for the attention half and for the MLP half of the block.
            m = _lin(sc, p, b + "adaLN_modulation.1")
            s1, c1, g1, s2, c2, g2 = [v[:, None] for v in mx.split(m, 6, axis=-1)]  # shift, scale, gate (x2); [:, None] broadcasts over tokens
            h = _ln(x) * (1 + c1) + s1  # modulate(norm1(x), shift, scale)
            # qkv: (N, T, 3*dim) -> (3, N, heads, T, head_dim)
            qkv = _lin(h, p, b + "attn.qkv").reshape(N, -1, 3, H, D).transpose(2, 0, 3, 1, 4)
            a = mx.fast.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=D ** -0.5)
            a = a.transpose(0, 2, 1, 3).reshape(N, -1, self.dim)  # merge heads back to (N, T, dim)
            x = x + g1 * _lin(a, p, b + "attn.proj")  # gated residual
            h = _ln(x) * (1 + c2) + s2
            h = _lin(h, p, b + "mlp.fc1")
            h = 0.5 * h * (1 + mx.tanh(math.sqrt(2 / math.pi) * (h + 0.044715 * h ** 3)))  # GELU, tanh approximation (as in SiT)
            x = x + g2 * _lin(h, p, b + "mlp.fc2")
        # FinalLayer: one more adaLN modulation (shift, scale only), then a linear map from each token to P*P*out_ch values.
        m = _lin(sc, p, "final_layer.adaLN_modulation.1")
        s, c_ = [v[:, None] for v in mx.split(m, 2, axis=-1)]
        x = _lin(_ln(x) * (1 + c_) + s, p, "final_layer.linear")
        # Unpatchify, the inverse of the patchify above: (N, g, g, P, P, C) -> (N, C, g, P, g, P) -> (N, C, 32, 32).
        x = x.reshape(N, g, g, P, P, self.out_ch).transpose(0, 5, 1, 3, 2, 4)
        x = x.reshape(N, self.out_ch, g * P, g * P).astype(mx.float32)
        return x[:, : self.in_ch] if self.learn_sigma else x  # drop the sigma channels, as SiT.forward does

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Same as sit/models.py: CFG on the first 3 channels only, duplicated batch."""
        half = x[: x.shape[0] // 2]
        out = self(mx.concatenate([half, half], axis=0), t, y)
        eps, rest = out[:, :3], out[:, 3:]
        cond, uncond = mx.split(eps, 2, axis=0)
        he = uncond + cfg_scale * (cond - uncond)
        return mx.concatenate([mx.concatenate([he, he], axis=0), rest], axis=1)
