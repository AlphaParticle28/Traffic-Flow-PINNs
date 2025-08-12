import torch
import torch.nn as nn

class TrafficFlowForLWR_PINN(nn.Module):
    def __init__(self, hidden_layers=8, neurons_per_layer=20):
        super(TrafficFlowForLWR_PINN, self).__init__()
        
        # Input: (x, t) -> Output: density ρ(x,t)
        layers = []
        layers.append(nn.Linear(2, neurons_per_layer))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons_per_layer, 1))  # Output layer

        # Create sequential model
        self.network = nn.Sequential(*layers)

        # Xavier initialization for all Linear layers
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)

class TrafficFlowForARZ_PINN(nn.Module):
    def __init__(self, hidden_layers=8, neurons_per_layer=20):
        super(TrafficFlowForARZ_PINN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(2, neurons_per_layer))  # Input: (x, t)
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(neurons_per_layer, 2))  # Output: (ρ, u)
        
        self.network = nn.Sequential(*layers)
        
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)

class FDLearner(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        # keep it small (paper uses a small NN)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, rho):
        # rho shape: (N,1)
        if rho.dim() == 1: rho = rho.unsqueeze(1)
        return self.net(rho)  # (N,1) -> Q
    