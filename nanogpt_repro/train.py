import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader

from data.dataloader import TinyShakespeare
from model import GPT
from config import Config


class Trainer:
    def __init__(
            self,
            trainer_cfg: Config,
            dataloader: DataLoader,
            model: nn.Module
    ):
        self.cfg = trainer_cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = AdamW(model.parameters(), lr=trainer_cfg.lr)

    def fit(self):
        losses = []
        for step, (x, y) in enumerate(self.dataloader):
            x = x.long()
            y = y.long()
            x = x.to(self.cfg.device, non_blocking=True)
            y = y.to(self.cfg.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            y_hat = model(x)
            loss = F.cross_entropy(y_hat.view(-1, self.cfg.V), y.view(-1))
            loss.backward()
            self.optimizer.step()
            if step % 10 == 0:
                losses.append(loss.item())
                # print(torch.cuda.memory_summary())


if __name__ == "__main__":
    cfg = Config()
    model = GPT(cfg)

    model.to(cfg.device)
    model.train()

    dataset = TinyShakespeare(cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.B, shuffle=True, pin_memory=True)
    trainer = Trainer(cfg, dataloader, model)
    trainer.fit()

