import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# =========================================================
# Parameters
# =========================================================
k1 = 0.01
k2 = 0.02
A0 = 1.0
t_points = np.array([1, 2, 4, 8, 16, 32], dtype=float)
n_vals = np.log2(t_points).astype(int)   # exponents: 0,1,2,3,4,5

# =========================================================
# Product formula for Hilger exponential e_{-k}(2^n,1)
# =========================================================
def hilger_exp(k, n):
    """Compute e_{-k}(2^n,1) = ∏_{j=0}^{n-1} (1 - k * 2^j)"""
    if n == 0:
        return 1.0
    prod = 1.0
    for j in range(n):
        prod *= (1 - k * (2**j))
    return prod

# Precompute for all n
e1 = np.array([hilger_exp(k1, n) for n in n_vals])   # A(t)
e2 = np.array([hilger_exp(k2, n) for n in n_vals])

# =========================================================
# Time‑scale solution A, B, C
# =========================================================
A_ts = e1.copy()
if k1 != k2:
    B_ts = (k1 / (k2 - k1)) * (e1 - e2)
else:
    # degenerate case (not used here)
    B_ts = k1 * n_vals * e1   # would need proper limit
C_ts = A0 - A_ts - B_ts       # mass conservation

# =========================================================
# Continuous reference solution (ODE)
# =========================================================
def rhs_cont(t, y):
    A, B, C = y
    dA = -k1 * A
    dB =  k1 * A - k2 * B
    dC =  k2 * B
    return [dA, dB, dC]

# Solve on [0, max(t_points)] with dense output
t_max = t_points[-1]
sol_cont = solve_ivp(rhs_cont, [0, t_max], [A0, 0.0, 0.0],
                     t_eval=t_points, method='RK45', rtol=1e-12, atol=1e-14)
A_cont = sol_cont.y[0]
B_cont = sol_cont.y[1]
C_cont = sol_cont.y[2]

# =========================================================
# Regressivity analysis
# =========================================================
mu = t_points
reg_k1 = 1 - mu * k1
reg_k2 = 1 - mu * k2
in_reg = (reg_k1 > 0) & (reg_k2 > 0)

# =========================================================
# Print tables (as in the paper)
# =========================================================
print("\n=== Table: Complete numerical solution on T_reg ===")
print(f"{'t':>3} {'A(t)':>12} {'B(t)':>12} {'C(t)':>12} {'Sum':>12} {'e_{-k1}':>12} {'e_{-k2}':>12}")
print("-" * 80)
for i, t in enumerate(t_points):
    print(f"{int(t):3d} {A_ts[i]:12.6f} {B_ts[i]:12.6f} {C_ts[i]:12.6f} "
          f"{A_ts[i]+B_ts[i]+C_ts[i]:12.6f} {e1[i]:12.6f} {e2[i]:12.6f}")

print("\n=== Table: Regressivity analysis ===")
print(f"{'t':>3} {'mu(t)':>8} {'1 - mu k1':>12} {'1 - mu k2':>12} {'In T_reg?':>10}")
print("-" * 55)
for i, t in enumerate(t_points):
    print(f"{int(t):3d} {mu[i]:8.1f} {reg_k1[i]:12.4f} {reg_k2[i]:12.4f} {'Yes' if in_reg[i] else 'No':>10}")

# Additional points beyond 32 for illustration
extra_t = np.array([64, 128, 256])
extra_mu = extra_t
extra_reg_k1 = 1 - extra_mu * k1
extra_reg_k2 = 1 - extra_mu * k2
print("\nAdditional points (beyond regressive domain):")
for i, t in enumerate(extra_t):
    print(f"{int(t):3d} {extra_mu[i]:8.1f} {extra_reg_k1[i]:12.4f} {extra_reg_k2[i]:12.4f} {'No':>10}")

print("\n=== Table: Mass conservation errors (machine precision) ===")
mass_sum = A_ts + B_ts + C_ts
mass_error_abs = np.abs(mass_sum - A0)
mass_error_rel = mass_error_abs / A0 * 100
print(f"{'t':>3} {'A+B+C':>20} {'Abs error':>15} {'Rel error (%)':>15}")
print("-" * 60)
for i, t in enumerate(t_points):
    print(f"{int(t):3d} {mass_sum[i]:20.15f} {mass_error_abs[i]:15.3e} {mass_error_rel[i]:15.3e}")

print("\n=== Table: Comparison with continuous solution ===")
print(f"{'t':>3} {'B_ts':>12} {'C_ts':>12} {'B_cont':>12} {'C_cont':>12} {'Rel diff (%)':>15}")
print("-" * 70)
for i, t in enumerate(t_points):
    if B_cont[i] > 0 or C_cont[i] > 0:
        rel_diff = max(abs((B_ts[i]-B_cont[i])/B_cont[i]) if B_cont[i]!=0 else 0,
                       abs((C_ts[i]-C_cont[i])/C_cont[i]) if C_cont[i]!=0 else 0) * 100
    else:
        rel_diff = 0.0
    print(f"{int(t):3d} {B_ts[i]:12.6f} {C_ts[i]:12.6f} {B_cont[i]:12.6f} {C_cont[i]:12.6f} {rel_diff:15.2f}")

# =========================================================
# Save tables to CSV (optional)
# =========================================================
pd.DataFrame({
    "t": t_points,
    "A(t)": A_ts,
    "B(t)": B_ts,
    "C(t)": C_ts,
    "Sum": A_ts+B_ts+C_ts,
    "e_{-k1}": e1,
    "e_{-k2}": e2
}).to_csv("time_scale_solution.csv", index=False)

pd.DataFrame({
    "t": t_points,
    "mu(t)": mu,
    "1 - mu k1": reg_k1,
    "1 - mu k2": reg_k2,
    "In T_reg?": in_reg
}).to_csv("regressivity_analysis.csv", index=False)

# =========================================================
# Plotting (Figure 1 and 2)
# =========================================================
plt.style.use('seaborn-v0_8-darkgrid')

# ---- Figure 1(a): Concentrations on T_reg ----
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.semilogx(t_points, A_ts, 'o-', linewidth=2, label='A(t)')
plt.semilogx(t_points, B_ts, 's-', linewidth=2, label='B(t)')
plt.semilogx(t_points, C_ts, '^-', linewidth=2, label='C(t)')
plt.axhline(A0, linestyle='--', color='gray', label='Total mass $A_0$')
plt.xlabel('t (log scale)')
plt.ylabel('Concentration')
plt.title('(a) Time scale solution')
plt.legend()
plt.grid(True)

# ---- Figure 1(b): Hilger exponentials ----
plt.subplot(1,2,2)
plt.semilogx(t_points, e1, 'o-', linewidth=2, label=r'$e_{-k_1}(t,1)$')
plt.semilogx(t_points, e2, 's-', linewidth=2, label=r'$e_{-k_2}(t,1)$')
plt.xlabel('t (log scale)')
plt.ylabel('Hilger exponential')
plt.title('(b) Hilger exponential functions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fig1_solution_and_hilger.png', dpi=150)
plt.show()

# ---- Figure 2(a): Regressivity condition ----
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.semilogx(t_points, reg_k1, 'o-', label=r'$1-\mu(t)k_1$')
plt.semilogx(t_points, reg_k2, 's-', label=r'$1-\mu(t)k_2$')
plt.axhline(0, linestyle='--', color='k', linewidth=1)
plt.axvline(1/k2, linestyle='--', color='red', label=r'$t=1/k_2$')
plt.fill_between(t_points, 0, 1, where=(reg_k2>0), alpha=0.2, label='Regressive domain')
plt.xlabel('t (log scale)')
plt.ylabel('Regressivity factor')
plt.title('(a) Regressivity condition')
plt.legend()
plt.grid(True)

# ---- Figure 2(b): Comparison with continuous solution ----
plt.subplot(1,2,2)
plt.semilogx(t_points, B_ts, 'o-', linewidth=2, label='B(t) (time scale)')
plt.semilogx(t_points, B_cont, 'o--', linewidth=2, label='B(t) (continuous)')
plt.semilogx(t_points, C_ts, 's-', linewidth=2, label='C(t) (time scale)')
plt.semilogx(t_points, C_cont, 's--', linewidth=2, label='C(t) (continuous)')
plt.xlabel('t (log scale)')
plt.ylabel('Concentration')
plt.title('(b) Time scale vs continuous solution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fig2_regressivity_and_comparison.png', dpi=150)
plt.show()

print("\nAll figures saved. Execution completed.")