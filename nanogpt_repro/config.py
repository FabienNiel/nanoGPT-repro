from dataclasses import dataclass
import torch


@dataclass
class Config:
    ## Model
    D: int = 768  # model embedding dim
    V: int = 50257  # vocabulary size
    L: int = 12  # number of layers
    max_seq_len: int = 1024  # max context length
    N: int = 8  # number of heads
    B: int = 1  # batch size
    dropout_rate_attn_c_proj: float = 0.4
    dropout_rate_attn_weights: float = 0.4
    dropout_rate_mlp_c_proj: float = 0.4

    ## Training
    lr: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_steps: int = 1000
    data_path: str = 'data/tiny_shakespeare'

    @property
    def H(self):
        # head dimension
        return self.D // self.N

    @property
    def F(self):
        # number of hidden dimensions in the MLP part
        return 4 * self.D

