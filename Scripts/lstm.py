import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # Take last time step output
        return out