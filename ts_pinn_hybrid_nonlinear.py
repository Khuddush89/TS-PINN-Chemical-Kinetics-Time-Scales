from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# PARAMETERS (same as successful run)
# =========================================================

k1 = 0.05
k2 = 0.01
k3 = 0.01

A0 = 1.0
B0 = 0.5
C0 = 0.2

M0 = A0 + B0 + C0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# REFERENCE SOLUTION
# =========================================================

def rhs(t, y):
    A, B, C = y
    return [
        -k1*A + k3*B*C,
        k1*A - k2*B**2 - k3*B*C,
        k2*B**2
    ]

t_cont = np.linspace(0, 10, 1000)

sol = solve_ivp(
    rhs,
    [0, 10],
    [A0, B0, C0],
    t_eval=t_cont,
    rtol=1e-10,
    atol=1e-12
)

A_cont, B_cont, C_cont = sol.y

A_prev = A_cont[-1]
B_prev = B_cont[-1]
C_prev = C_cont[-1]

t_disc = np.arange(11, 51)

A_disc = []
B_disc = []
C_disc = []

for _ in t_disc:
    dA = -k1*A_prev + k3*B_prev*C_prev
    dB =  k1*A_prev - k2*B_prev**2 - k3*B_prev*C_prev
    dC =  k2*B_prev**2

    A_prev += dA
    B_prev += dB
    C_prev += dC

    A_disc.append(A_prev)
    B_disc.append(B_prev)
    C_disc.append(C_prev)

A_disc = np.array(A_disc)
B_disc = np.array(B_disc)
C_disc = np.array(C_disc)

t_ref = np.concatenate([t_cont, t_disc])
A_ref = np.concatenate([A_cont, A_disc])
B_ref = np.concatenate([B_cont, B_disc])
C_ref = np.concatenate([C_cont, C_disc])

# =========================================================
# NETWORK (50 neurons, 3 layers – proven working)
# =========================================================

class TSPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
        )

    def forward(self, x):
        return self.net(x)

model = TSPINN().to(device)

# =========================================================
# RESIDUALS
# =========================================================

def residual_cont(tc):
    tc = tc.clone().detach().requires_grad_(True)
    out = model(tc)
    A = out[:,0:1]
    B = out[:,1:2]
    C = out[:,2:3]

    dA = torch.autograd.grad(A.sum(), tc, create_graph=True)[0]
    dB = torch.autograd.grad(B.sum(), tc, create_graph=True)[0]
    dC = torch.autograd.grad(C.sum(), tc, create_graph=True)[0]

    RA = dA + k1*A - k3*B*C
    RB = dB - k1*A + k2*B**2 + k3*B*C
    RC = dC - k2*B**2
    return RA, RB, RC, A, B, C

def residual_disc(td):
    out = model(td)
    out_next = model(td + 1)

    A = out[:,0:1]
    B = out[:,1:2]
    C = out[:,2:3]

    A1 = out_next[:,0:1]
    B1 = out_next[:,1:2]
    C1 = out_next[:,2:3]

    RA = (A1 - A) + k1*A - k3*B*C
    RB = (B1 - B) - k1*A + k2*B**2 + k3*B*C
    RC = (C1 - C) - k2*B**2
    return RA, RB, RC, A, B, C

# =========================================================
# LOSS FUNCTION (no equilibrium penalty)
# =========================================================

def loss_function(tc, td):
    RAc, RBc, RCc, Ac, Bc, Cc = residual_cont(tc)
    RAd, RBd, RCd, Ad, Bd, Cd = residual_disc(td)

    loss_res = 100 * (
        torch.mean(RAc**2) + torch.mean(RBc**2) + torch.mean(RCc**2) +
        torch.mean(RAd**2) + torch.mean(RBd**2) + torch.mean(RCd**2)
    )

    y0 = model(torch.tensor([[0.0]], dtype=torch.float32, device=device))
    loss_ic = (y0[0,0] - A0)**2 + (y0[0,1] - B0)**2 + (y0[0,2] - C0)**2

    Aall = torch.cat([Ac, Ad])
    Ball = torch.cat([Bc, Bd])
    Call = torch.cat([Cc, Cd])

    loss_cons = torch.mean((Aall + Ball + Call - M0)**2)

    loss_pos = (
        torch.mean(torch.relu(-Aall)) +
        torch.mean(torch.relu(-Ball)) +
        torch.mean(torch.relu(-Call))
    )

    total_loss = 10*loss_ic + loss_res + 1000*loss_cons + loss_pos
    return total_loss

# =========================================================
# TRAINING DATA
# =========================================================

tc = torch.linspace(0, 10, 200).view(-1, 1).to(device)   # 200 collocation points
td = torch.tensor(np.arange(11, 50), dtype=torch.float32).view(-1, 1).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

epochs = 15000
loss_history = []

# =========================================================
# TRAINING
# =========================================================

start = time.time()
print("Training started...")

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = loss_function(tc, td)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss = {loss.item():.6e}")

runtime = time.time() - start
print("\nTraining completed.")

# =========================================================
# PREDICTIONS
# =========================================================

with torch.no_grad():
    t_cont_eval = torch.linspace(0, 10, 500).view(-1, 1).to(device)
    pred_cont = model(t_cont_eval).cpu().numpy()

    t_disc_eval = torch.tensor(np.arange(11, 51), dtype=torch.float32).view(-1, 1).to(device)
    pred_disc = model(t_disc_eval).cpu().numpy()

t_pred = np.concatenate([t_cont_eval.cpu().numpy().flatten(), t_disc_eval.cpu().numpy().flatten()])
A_pred = np.concatenate([pred_cont[:,0], pred_disc[:,0]])
B_pred = np.concatenate([pred_cont[:,1], pred_disc[:,1]])
C_pred = np.concatenate([pred_cont[:,2], pred_disc[:,2]])

# =========================================================
# ERROR ANALYSIS
# =========================================================

fA = interp1d(t_ref, A_ref, fill_value="extrapolate")
fB = interp1d(t_ref, B_ref, fill_value="extrapolate")
fC = interp1d(t_ref, C_ref, fill_value="extrapolate")

A_exact = fA(t_pred)
B_exact = fB(t_pred)
C_exact = fC(t_pred)

mse_A = np.mean((A_pred - A_exact)**2)
mse_B = np.mean((B_pred - B_exact)**2)
mse_C = np.mean((C_pred - C_exact)**2)

linf_A = np.max(np.abs(A_pred - A_exact))
linf_B = np.max(np.abs(B_pred - B_exact))
linf_C = np.max(np.abs(C_pred - C_exact))

mass_error = np.max(np.abs(A_pred + B_pred + C_pred - M0))

# =========================================================
# REPORT
# =========================================================

print("\n=== ERROR METRICS ===")
print(f"MSE_A = {mse_A:.6e}")
print(f"MSE_B = {mse_B:.6e}")
print(f"MSE_C = {mse_C:.6e}")
print(f"Linf_A = {linf_A:.6e}")
print(f"Linf_B = {linf_B:.6e}")
print(f"Linf_C = {linf_C:.6e}")
print(f"Mass Error = {mass_error:.6e}")

print("\n=== HARDWARE ===")
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("CPU execution")
print(f"Epochs : {epochs}")
print(f"Runtime (s): {runtime:.2f}")

# =========================================================
# SAVE CSV
# =========================================================

pd.DataFrame({
    "Variable": ["A", "B", "C"],
    "MSE": [mse_A, mse_B, mse_C],
    "Linf": [linf_A, linf_B, linf_C]
}).to_csv("error_metrics.csv", index=False)

pd.DataFrame({
    "Metric": ["Epochs", "Runtime (s)", "Mass Error"],
    "Value": [epochs, runtime, mass_error]
}).to_csv("performance_metrics.csv", index=False)

# =========================================================
# PLOTS
# =========================================================

plt.figure(figsize=(8,5))
plt.plot(t_ref, A_ref, 'k--', linewidth=1.5, label='A ref')
plt.plot(t_ref, B_ref, 'k-.', linewidth=1.5, label='B ref')
plt.plot(t_ref, C_ref, 'k:', linewidth=1.5, label='C ref')
plt.scatter(t_pred, A_pred, s=8, alpha=0.6, label='A PINN')
plt.scatter(t_pred, B_pred, s=8, alpha=0.6, label='B PINN')
plt.scatter(t_pred, C_pred, s=8, alpha=0.6, label='C PINN')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.title('TS-PINN vs Reference Solution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('solution_comparison.png', dpi=150)
plt.show()

plt.figure(figsize=(8,5))
mass_pred = A_pred + B_pred + C_pred
plt.plot(t_pred, mass_pred, 'b-', linewidth=1.5, label='A+B+C (PINN)')
plt.axhline(M0, color='r', linestyle='--', linewidth=2, label=f'M0 = {M0}')
plt.xlabel('t')
plt.ylabel('Total mass')
plt.title('Mass Conservation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mass_conservation.png', dpi=150)
plt.show()

plt.figure(figsize=(8,5))
plt.semilogy(loss_history, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()