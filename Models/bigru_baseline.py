import torch
import torch.nn as nn


class BiGRUBaseline(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, y_win, x0):
        batch_size, _, TAU = y_win.shape
        y_seq = y_win.permute(0, 2, 1)
        x0_seq = x0.unsqueeze(1).expand(-1, TAU, -1)
        inp = torch.cat([y_seq, x0_seq], dim=-1)
        out, _ = self.bigru(inp)
        return self.head(out).squeeze(-1)
