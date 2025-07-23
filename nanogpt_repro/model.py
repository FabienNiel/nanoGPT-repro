import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2LMHeadModel
from typing import List

from .config import Config


class AttentionBlock(nn.Module):
    def __init__(self, cfg: Config):
        super(AttentionBlock, self).__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.D, 3 * cfg.D)
        self.c_proj = nn.Linear(cfg.D, cfg.D)
        self.dropout_c_proj = nn.Dropout(cfg.dropout_rate_attn_c_proj)
        self.dropout_attn_weights = nn.Dropout(cfg.dropout_rate_attn_weights)
        mask = torch.ones((cfg.max_seq_len, cfg.max_seq_len), dtype=torch.bool).tril()
        self.register_buffer('mask', mask, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
            use_kv_cache: bool = True
    ):
        if x.ndim != 3:
            raise ValueError(
                f'x should have shape (B, T, D) with B batch size, T sequence length and D embedding dim. Got {x.shape}'
            )
        B, T_new, D = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        H = self.cfg.D // self.cfg.N
        q = q.view((B, T_new, self.cfg.N, H))
        k = k.view((B, T_new, self.cfg.N, H))
        v = v.view((B, T_new, self.cfg.N, H))
        if kv_cache is not None:
            old_k, old_v = kv_cache
            k = torch.cat((old_k, k), dim=1)
            v = torch.cat((old_v, v), dim=1)
        T_total = k.shape[1]
        scale = H ** 0.5
        unnorm_weights = torch.einsum('btnh, bsnh -> btsn', q, k) / scale
        past_len = k.size(1) - T_new
        mask = torch.logical_not(self.mask.to(device=x.device)[None, past_len:past_len + T_new, :T_total, None])
        unnorm_weights = torch.masked_fill(unnorm_weights, mask, float("-inf"))
        norm_weights = self.dropout_attn_weights(F.softmax(unnorm_weights, dim=2))
        weighted_sum = torch.einsum('btsn, bsnh -> btnh', norm_weights, v).reshape(B, T_new, D)
        output = self.dropout_c_proj(self.c_proj(weighted_sum))
        present_kv = (k, v)
        return output, present_kv


class MLPBlock(nn.Module):
    def __init__(self, cfg: Config):
        super(MLPBlock, self).__init__()
        self.c_fc = nn.Linear(cfg.D, cfg.F)
        self.c_proj = nn.Linear(cfg.F, cfg.D)
        self.dropout = nn.Dropout(cfg.dropout_rate_mlp_c_proj)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(
            self,
            cfg: Config
    ):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(cfg.D)
        self.attn = AttentionBlock(cfg)
        self.ln_2 = nn.LayerNorm(cfg.D)
        self.mlp = MLPBlock(cfg)

    def forward(
            self,
            x: torch.Tensor,
            kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ):
        out_attn, present_kv = self.attn(self.ln_1(x), kv_cache)
        x = x + out_attn
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class Transformer(nn.Module):
    def __init__(
            self,
            cfg: Config
    ):
        super(Transformer, self).__init__()
        self.wte = nn.Embedding(cfg.V, cfg.D)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.D)
        self.h = nn.ModuleList(
            [
                TransformerBlock(cfg) for _ in range(cfg.L)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.D)

    def forward(
            self,
            x: torch.LongTensor,
            kv_caches: List[tuple[torch.Tensor, torch.Tensor]] | List[None] = None
    ):
        if x.ndim != 2:
            raise ValueError(
                f"Expected input of shape (B, T) where B is the batch size and T the sequence length, got {x.shape}"
            )
        if x.dtype != torch.long:
            raise TypeError(
                f"x must be a torch.LongTensor, got {x.dtype}"
            )
        if kv_caches is None:
            kv_caches = [None] * len(self.h)

        B, T_new = x.shape
        past_len = 0
        if kv_caches[0] is not None:  # kv_caches is now a list
            past_len = kv_caches[0][0].size(1)
        pos = torch.arange(past_len, past_len + T_new,
                           device=x.device).unsqueeze(0)
        x = self.wte(x) + self.wpe(pos)
        present_kvs = []
        for layer, kv_cache in zip(self.h, kv_caches):
            x, present_kv = layer(x, kv_cache)
            present_kvs.append(present_kv)
        return self.ln_f(x), present_kvs


class GPT(nn.Module):
    def __init__(
            self,
            cfg: Config
    ):
        super(GPT, self).__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)
        self.lm_head = nn.Linear(cfg.D, cfg.V, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, TransformerBlock):
            nn.init.normal_(module.attn.c_proj.weight, mean=0.0, std=0.02 / self.cfg.L ** 0.5)
            nn.init.normal_(module.mlp.c_proj.weight, mean=0.0, std=0.02 / self.cfg.L ** 0.5)

    def forward(
            self,
            x: torch.Tensor,
            kv_caches: List[tuple[torch.Tensor, torch.Tensor]] | List[None] = None
    ):
        x, present_kvs = self.transformer(x, kv_caches)
        x = self.lm_head(x)
        return x, present_kvs

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(L=12, N=12, D=768),  # 124M params
            'gpt2-medium':  dict(L=24, N=16, D=1024),  # 350M params
            'gpt2-large':   dict(L=36, N=20, D=1280),  # 774M params
            'gpt2-xl':      dict(L=48, N=25, D=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['V'] = 50257 # always 50257 for GPT model checkpoints
        config_args['max_seq_len'] = 1024 # always 1024 for GPT model checkpoints
        # config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, temperature=1.0, num_token_to_generate=30):
        # build cache for the prompt
        kv_caches = [None] * self.cfg.L
        logits, kv_caches = self(idx, kv_caches)

        for _ in range(num_token_to_generate):
            # ── sample next token ───────────────────────────────────────────
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append to running sequence
            idx = torch.cat([idx, idx_next], dim=1)

            # ── forward **only the new token** using the cache ─────────────
            logits, kv_caches = self(idx_next, kv_caches)  # <─ FIX
            #    (was self(idx, kv_caches)
        return idx
