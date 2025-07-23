import tiktoken
import torch

from nanogpt_repro.model import GPT
from nanogpt_repro.config import Config


if __name__ == "__main__":
    cfg = Config()
    tokenizer = tiktoken.encoding_for_model("gpt2")
    model = GPT.from_pretrained("gpt2").eval().to(cfg.device)

    prompt_ids = tokenizer.encode("Hello, I am")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.device)

    out = model.generate(idx, temperature=0.8, num_token_to_generate=80)
    print(tokenizer.decode(out[0].tolist()))
