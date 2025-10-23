import torch
import torch.nn as nn

class VanillaMLP(nn.Module):
    def __init__(self, hidden_layers=8, neurons_per_layer=20, output_dim=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, neurons_per_layer))  # Input: (x, t)
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons_per_layer, output_dim))  # Output: (ρ, u)
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)