import os

import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
from nanogpt_repro.config import Config


class TinyShakespeare(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        tokenizer = tiktoken.encoding_for_model('gpt2')
        with open(os.path.join(os.getcwd(), cfg.data_path), 'r') as f:
            text_data = f.read()
        tokenized_data = torch.as_tensor(tokenizer.encode(text_data))
        divisible_length = (round(tokenized_data.shape[0]/(cfg.max_seq_len+1))+1)*(cfg.max_seq_len+1)
        padding = torch.zeros(divisible_length-tokenized_data.shape[0])
        tokenized_data = torch.cat((tokenized_data, padding), dim=0)
        self.data = tokenized_data.reshape((-1, cfg.max_seq_len+1))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x, y = self.data[idx, :-1], self.data[idx, 1:]
        return x, y


if __name__ == "__main__":
    cfg = Config()
    dataset = TinyShakespeare(cfg=cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.B, shuffle=True, pin_memory=True)
