import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPT2LMHeadModel

from config import Config


class AttentionBlock(nn.Module):
    def __init__(self, cfg: Config):
        super(AttentionBlock, self).__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.D, 3 * cfg.D)
        self.c_proj = nn.Linear(cfg.D, cfg.D)
        self.dropout_c_proj = nn.Dropout(cfg.dropout_rate_attn_c_proj)
        self.dropout_attn_weights = nn.Dropout(cfg.dropout_rate_attn_weights)
        mask = torch.ones((cfg.max_seq_len, cfg.max_seq_len)).tril()
        self.register_buffer('mask', mask, persistent=False)
        self.kv_cache = None

    def forward(self, x: torch.Tensor, use_kv_cache: bool = True) -> torch.Tensor:
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
        if self.kv_cache is not None:
            k = torch.cat((self.kv_cache[0], k), dim=1)
            v = torch.cat((self.kv_cache[1], v), dim=1)
        self.kv_cache = torch.stack((k, v), dim=0)
        unnorm_weights = torch.einsum('btnh, bsnh -> btsn', q, k) / H ** 0.5
        unnorm_weights = torch.masked_fill(unnorm_weights, torch.logical_not(self.mask.to(dtype=x.dtype, device=x.device)[None, :T_new, :T_new, None]),
                                           float("-inf"))
        norm_weights = self.dropout_attn_weights(F.softmax(unnorm_weights, dim=2))
        weighted_sum = torch.einsum('btsn, bsnh -> btnh', norm_weights, v).reshape(B, T_new, D)
        output = self.dropout_c_proj(self.c_proj(weighted_sum))
        return output


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected input of shape (B, T) where B is the batch size and T the sequence length, got {x.shape}"
            )
        if x.dtype != torch.long:
            raise TypeError(
                f"x must be a torch.LongTensor, got {x.dtype}"
            )
        B, T = x.shape
        positions = torch.arange(T).unsqueeze(0).to(dtype=x.dtype, device=x.device)
        x = self.wte(x) + self.wpe(positions)
        for layer in self.h:
            x = layer(x)
        return self.ln_f(x)


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

    def forward(self, x):
        x = self.transformer(x)
        x = self.lm_head(x)
        return x

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
    def generate(self, idx, temperature=1, num_token_to_generate: int = 30):
      assert len(idx.shape) == 2
      for _ in range(num_token_to_generate):
        # if the sequence context is growing too long we must crop it at block_size
        # forward the model to get the logits for the index in the sequence
        logits = self(idx)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
      return idx


if __name__ == "__main__":
    cfg = Config()
    gpt = GPT(cfg)
    test_input = torch.randint(0, cfg.V-1, (cfg.B, cfg.max_seq_len, cfg.D))
    test_output = gpt(test_input)
    print(test_input.shape, test_output.shape)
    # gpt_state_dict = gpt.state_dict()
    # for k, v in gpt_state_dict.items():
    #     print(k, v.shape)
