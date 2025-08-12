import torch
from Scripts.pinn_model import FDLearner
fd_model = FDLearner()

def lwr_pde_residual(model, x, t, fd_model=fd_model, v_max=1.02, rho_max=1.13, eps=0.005):
    """
    Compute LWR equation residual: ∂ρ/∂t + ∂(ρv)/∂x−ε∂(∂ρ/∂x)/∂x = 0
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    rho = model(x, t)
    q = fd_model(rho) 
    
    # Compute derivatives using autograd
    rho_t = torch.autograd.grad(rho.sum(), t, create_graph=True)[0]
    
    # # Fundamental diagram: v = v_max * (1 - ρ/ρ_max)
    # v = v_max * (1 - rho / rho_max)
    
    # # Flow: q = ρ * v
    # q = rho * v ideally

    q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]

    rho_x = torch.autograd.grad(rho.sum(), x, create_graph=True)[0]
    rho_xx = torch.autograd.grad(rho_x.sum(), x, create_graph=True)[0]

    # PDE residual
    pde_residual = rho_t + q_x - eps * rho_xx
    return pde_residual

def arz_pde_residual(model, x, t, fd_model=fd_model, v_max=1.02, rho_max=1.13, tau=0.02):
    """
    Compute ARZ equation residual: 
    1] ∂ρ/∂t + ∂(ρu)/∂x = 0 = f1 # first residual function

    Ueq(ρ) = V_max(1 - ρ/ρ_max) # Flow at density ρ
    h(ρ) = Ueq(0) - Ueq(ρ) = V_max - V_max(1 - ρ/ρ_max) = V_max * ρ/ρ_max     # Traffic Pressure at ρ

    2] ∂(u + h(ρ))/ ∂t + u * ∂(u + h(ρ))/∂x - (Ueq(ρ) - u) / τ = 0 = f2 # second residual function

    #### Dealt with in a different function below ####
    3] ρ(t, 0) = ρ(t, 1) # periodic boundary condition
    4] u(t, 0) = u(t, 1) # periodic boundary condition

    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    rho, u = model(x, t).split(1, dim=1)
    
    # Compute derivatives using autograd
    rho_t = torch.autograd.grad(rho.sum(), t, create_graph=True)[0]
    
    # # Fundamental diagram: v = v_max * (1 - ρ/ρ_max)
    # Ueq = v_max * (1 - rho / rho_max) ideally
    Ueq = fd_model(rho)  # Using FD model to get Ueq from rho
    Ueq0 = fd_model(torch.zeros_like(rho))
    Q = rho * u
    Q_x = torch.autograd.grad(Q.sum(), x, create_graph=True)[0]
    physics_loss1 = rho_t + Q_x

    h = Ueq0 - Ueq
    v = u + h

    physics_loss2 = torch.autograd.grad(v.sum(), t, create_graph=True)[0] + u * torch.autograd.grad(v.sum(), x, create_graph=True)[0] - (Ueq - u) / tau
    #pde_residual = physics_loss1 + physics_loss2

    return physics_loss1, physics_loss2

def arz_boundary_loss(model, tb, device='cpu'):
    # tb: tensor of shape (Nb, 1) with times where we enforce boundary conditions
    tb = tb.reshape(-1, 1)  # ✅ ensure 2D
    x0 = torch.zeros_like(tb).to(device)
    x1 = torch.ones_like(tb).to(device)  # assuming domain normalized to [0, 1]

    rho0, u0 = model(x0, tb).split(1, dim=1)
    rho1, u1 = model(x1, tb).split(1, dim=1)

    loss_rho_bc = torch.mean((rho0 - rho1)**2)
    loss_u_bc   = torch.mean((u0 - u1)**2)

    return loss_rho_bc, loss_u_bc


def physics_loss_calculator(x, t, punn_model, beta1=1.0, beta2=1.0,
                            gamma1=1.0, gamma2=1.0, tau=0.02):
    # returns scalar total PDE + BC loss (already squared & mean'd)
    f1, f2 = arz_pde_residual(punn_model, x, t, tau=tau)  # f1,f2 shape (N,1)

    # boundary losses remain as before but needs punn_model outputs split
    bc1, bc2 = arz_boundary_loss(punn_model, t)  # keep as implemented (mean scalars)
    total = (beta1 * f1.pow(2).mean() +
             beta2 * f2.pow(2).mean() +
             gamma1 * bc1 +
             gamma2 * bc2)
    return total