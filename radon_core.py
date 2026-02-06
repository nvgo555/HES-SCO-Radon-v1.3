"""
HES-SCO RADON v1.3: GEOMETRIC INTELLIGENCE ARCHITECTURE
================================================================================
Author: Naveen George Sunny (Architect) & Gemini 3 Pro (Engineer)
Date:   February 6, 2026
Spec:   v1.3 (Post-Euclidean / Low-Compute / D=64)

DESCRIPTION:
This is the reference implementation of the Radon v1.3 architecture.
It replaces standard Deep Learning (Curve Fitting) with Geometric Manifold Mapping.
It features a Hybrid Topology (Euclidean x Hyperbolic x Spherical) and an 
Inference-Time Optimization Loop ("System 2 Thinking").

COMPONENTS:
1. THE MANIFOLD CAST: Log-Compressed projection to H^16, S^16, R^32.
2. SYSTEM 2 ENGINE: Riemannian Gradient Descent loop for latent reasoning.
3. GEOMETRIC MEMORY: SPD Matrices (Worldview) & Grassmannian Subspaces (Context).
4. POINCARE READOUT: Bounded projection ensuring OOD robustness.

DEPENDENCIES:
- torch
- geoopt (Riemannian Optimization)
- numpy, matplotlib
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# Auto-Install Geoopt if missing (Colab convenience)
try:
    import geoopt
except ImportError:
    import subprocess
    print("⚙️ Installing Geoopt...")
    subprocess.check_call(["pip", "install", "geoopt"])
    import geoopt

# Set Double Precision for Geometric Stability
torch.set_default_dtype(torch.float64)

# ==============================================================================
# SECTION 1: THE GEOMETRIC KERNEL (PHYSICS ENGINE)
# ==============================================================================

class RadonManifoldCast(nn.Module):
    """
    The Physical Kernel.
    Tears the input vector apart and casts it onto 3 distinct topologies.
    Features 'Logarithmic Compression' to handle infinite OOD noise.
    """
    def __init__(self):
        super().__init__()
        self.dim_E = 32
        self.dim_H = 16
        self.dim_S = 16
        
        # Manifolds
        self.manifold_H = geoopt.Lorentz(k=1.0, learnable=False) 
        self.manifold_S = geoopt.Sphere()
        
    def forward(self, x_raw):
        # A. Split
        raw_E, raw_H, raw_S = torch.split(x_raw, [self.dim_E, self.dim_H, self.dim_S], dim=-1)
        
        # B. Euclidean (Pass-through)
        z_E = raw_E 
        
        # C. Hyperbolic (Log-Compressed)
        # Physics: Forces input 'force' to scale logarithmically.
        # Input 5.0 -> 1.79 (Manageable) vs 5.0 (Massive)
        norm_H = raw_H.norm(p=2, dim=-1, keepdim=True)
        compression = torch.log1p(norm_H) / (norm_H + 1e-6)
        safe_raw_H = raw_H * compression
        
        # ExpMap: Tangent -> Manifold
        origin_H = self.manifold_H.origin(raw_H.shape[:-1] + (self.dim_H+1,)).to(x_raw.device)
        zeros = torch.zeros_like(safe_raw_H[..., :1])
        tangent_vector = torch.cat([zeros, safe_raw_H], dim=-1)
        z_H = self.manifold_H.expmap(origin_H, tangent_vector)
        
        # D. Spherical (Projected)
        z_S = self.manifold_S.projx(raw_S)
        
        return z_E, z_H, z_S

class IntrinsicMobiusLinear(nn.Module):
    """
    The Universal Manifold Operator.
    Linear Transformation respecting curved geometry via Tangent Space.
    """
    def __init__(self, manifold, in_features, out_features):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        origin = self.manifold.origin(x.shape).to(x.device)
        x_tan = self.manifold.logmap(origin, x)
        y_raw = self.linear(x_tan)
        # Force Tangency (Project u) - The "Law of Physics"
        y_tan = self.manifold.proju(origin, y_raw)
        y = self.manifold.expmap(origin, y_tan)
        return y

class EpiplexityGate(nn.Module):
    """
    The 'Feeling' of the Model.
    Measures geometric tension/surprise.
    """
    def __init__(self, dim_total=64):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(dim_total, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, z_E, z_H, z_S):
        state_flat = torch.cat([z_E, z_H, z_S], dim=-1)
        E = self.readout(state_flat)
        return E

# ==============================================================================
# SECTION 2: THE SYSTEM 2 ENGINE (THE GHOST)
# ==============================================================================

class RadonSystem2Engine(nn.Module):
    """
    The Cognitive Core.
    Implements the 'Thinking Loop' (Riemannian Gradient Descent on Latent State).
    """
    def __init__(self):
        super().__init__()
        self.caster = RadonManifoldCast()
        # Processors
        self.proc_H = IntrinsicMobiusLinear(self.caster.manifold_H, 17, 17)
        self.proc_S = IntrinsicMobiusLinear(self.caster.manifold_S, 16, 16)
        self.proc_E = nn.Linear(32, 32)
        # Monitor
        self.gate = EpiplexityGate(dim_total=32+17+16)
        
    def forward(self, x_raw, thinking_budget=30, lr=0.05):
        # 1. Reflex (System 1)
        z_E, z_H, z_S = self.caster(x_raw)
        z_E = self.proc_E(z_E)
        z_H = self.proc_H(z_H)
        z_S = self.proc_S(z_S)
        
        # 2. Detach for Thinking
        z_E_think = z_E.clone().detach().requires_grad_(True)
        z_H_think = z_H.clone().detach().requires_grad_(True)
        z_S_think = z_S.clone().detach().requires_grad_(True)
        
        trajectory = [] # For Visualization
        
        def capture_state(zh):
            # Log Poincare projection for viz
            t = zh[0, 0]; x = zh[0, 1:]
            zp = x / (1 + t)
            return zp[:2].detach().cpu().numpy() # Just 2 dims for plotting

        if thinking_budget > 0:
            trajectory.append(capture_state(z_H_think))

        # 3. The Loop (System 2)
        for t in range(thinking_budget):
            energy = self.gate(z_E_think, z_H_think, z_S_think)
            
            # Stop if calm
            if energy.mean() < 0.1: break
                
            # Gradient
            grad_E, grad_H, grad_S = torch.autograd.grad(
                energy.sum(), [z_E_think, z_H_think, z_S_think]
            )
            
            # Update (RGD)
            with torch.no_grad():
                z_E_think.sub_(lr * grad_E) # Euclidean
                
                rgrad_H = self.caster.manifold_H.egrad2rgrad(z_H_think, grad_H)
                z_H_think.data = self.caster.manifold_H.retr(z_H_think, -lr * rgrad_H)
                
                rgrad_S = self.caster.manifold_S.egrad2rgrad(z_S_think, grad_S)
                z_S_think.data = self.caster.manifold_S.retr(z_S_think, -lr * rgrad_S)
            
            # Reset & Log
            z_E_think.grad = None; z_H_think.grad = None; z_S_think.grad = None
            trajectory.append(capture_state(z_H_think))

        return (z_E_think, z_H_think, z_S_think), trajectory

# ==============================================================================
# SECTION 3: GEOMETRIC MEMORY
# ==============================================================================

class RadonMemoryCore(nn.Module):
    """
    Stores 'Knowledge' as Geometric Shapes.
    - Frame Operator (SPD): Covariance/Uncertainty Structure.
    - Task Switcher (Stiefel/Grassmannian): Context Subspaces.
    """
    def __init__(self, dim_H=16):
        super().__init__()
        self.dim_H = dim_H
        self.manifold_SPD = geoopt.SymmetricPositiveDefinite()
        self.frame_operator = geoopt.ManifoldParameter(
            torch.eye(dim_H).unsqueeze(0), manifold=self.manifold_SPD
        )
        self.manifold_St = geoopt.Stiefel()
        # Initialize Task Basis (Orthonormal)
        q, _ = torch.linalg.qr(torch.randn(1, dim_H, 2))
        self.task_basis = geoopt.ManifoldParameter(q, manifold=self.manifold_St)

    def update_worldview(self, z_batch, lr=0.01):
        # Geodesic update of covariance matrix
        z_cen = z_batch - z_batch.mean(0, keepdim=True)
        cov = (z_cen.t() @ z_cen) / (z_batch.shape[0] - 1 + 1e-8)
        cov = cov + 1e-5 * torch.eye(self.dim_H, device=z_batch.device)
        target = cov.unsqueeze(0)
        
        vec = self.manifold_SPD.logmap(self.frame_operator, target)
        self.frame_operator.data = self.manifold_SPD.expmap(self.frame_operator, lr * vec)

# ==============================================================================
# SECTION 4: THE UNIFIED AGENT & GOLIATH
# ==============================================================================

class RadonUnifiedAgent(nn.Module):
    """
    The Full Architecture (D=64).
    Integrates Kernel, Engine, Memory, and Poincare Readout.
    """
    def __init__(self):
        super().__init__()
        self.brain = RadonSystem2Engine()
        self.memory = RadonMemoryCore()
        
        # Readout: Maps 3 manifolds to decision
        # We project Lorentz -> Poincare before this layer
        self.readout = nn.Sequential(
            nn.Linear(32 + 16 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def lorentz_to_poincare(self, z_H):
        t = z_H[..., 0:1]; x = z_H[..., 1:]
        return x / (1 + t) # Guaranteed ||y|| < 1

    def forward(self, x, use_system2=True):
        (z_E, z_H, z_S), trace = self.brain(x, thinking_budget=30 if use_system2 else 0)
        
        z_P = self.lorentz_to_poincare(z_H) # Bounded
        z_E_bounded = torch.tanh(z_E)       # Bounded
        
        decision = self.readout(torch.cat([z_E_bounded, z_P, z_S], dim=-1))
        return decision, trace

class GoliathMLP(nn.Module):
    """ The Baseline: Standard Euclidean Network (D=1024). """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        return self.net(x), []

# ==============================================================================
# SECTION 5: TOOLS (AUDIT & VIZ)
# ==============================================================================

class RadonAuditToolkit:
    @staticmethod
    def run_audit(radon_model, goliath_model):
        print("\n📊 RADON v1.3 ENGINEERING AUDIT")
        print("============================================================")
        
        # 1. Complexity
        p_r = sum(p.numel() for p in radon_model.parameters())
        p_g = sum(p.numel() for p in goliath_model.parameters())
        mem_r = (p_r * 8) / 1048576; mem_g = (p_g * 8) / 1048576
        print(f"1. FOOTPRINT: Radon {mem_r:.4f} MB vs Goliath {mem_g:.4f} MB")
        print(f"   ➤ Compression: {mem_g/mem_r:.1f}x")
        
        # 2. Entropy
        F = radon_model.memory.frame_operator.detach()
        evals = torch.linalg.eigvalsh(F[0])
        prob = evals / evals.sum()
        entropy = -torch.sum(prob * torch.log(prob + 1e-9)).item()
        print(f"\n2. MEMORY ENTROPY: {entropy:.4f} nats (Status: {'CRYSTALLIZED' if entropy < 1.5 else 'HIGH'})")

class RadonVisualizer:
    @staticmethod
    def plot_thought_process(trajectory):
        if not trajectory: return
        traj = np.array(trajectory)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect('equal')
        
        circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--')
        ax.add_artist(circle)
        ax.text(0, 1.05, "Manifold Horizon", ha='center')
        
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.6, label='Thinking Path')
        ax.plot(traj[0, 0], traj[0, 1], 'ro', label='Panic (Start)')
        ax.plot(traj[-1, 0], traj[-1, 1], 'go', label='Logic (End)')
        
        ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
        ax.set_title("Visualizing System 2 (Poincaré Disk)")
        plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("⚔️ HES-SCO RADON v1.3: FINAL BENCHMARK PROTOCOL")
    
    radon = RadonUnifiedAgent().to(torch.double)
    goliath = GoliathMLP().to(torch.double)
    
    # 1. TRAIN (Clean Hierarchy)
    print("\n📚 Phase 1: Training (Clean Data)...")
    opt_r = torch.optim.Adam(radon.parameters(), lr=0.01)
    opt_g = torch.optim.Adam(goliath.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    x_train = torch.randn(50, 64)
    y_train = (x_train[:, 0] > x_train[:, 1]).double().unsqueeze(1)
    
    for _ in range(60):
        # Radon
        p_r, _ = radon(x_train, use_system2=False)
        l_r = criterion(p_r, y_train)
        opt_r.zero_grad(); l_r.backward(); opt_r.step()
        # Goliath
        p_g, _ = goliath(x_train)
        l_g = criterion(p_g, y_train)
        opt_g.zero_grad(); l_g.backward(); opt_g.step()
        
    print(f"   Loss -> Radon: {l_r.item():.4f} | Goliath: {l_g.item():.4f}")
    
    # 2. TEST (OOD Noise)
    print("\n🌪️ Phase 2: OOD Stress Test (Noise x5.0)...")
    x_test = torch.randn(20, 64) * 5.0
    y_test = (x_test[:, 0] > x_test[:, 1]).double().unsqueeze(1)
    
    pred_g, _ = goliath(x_test)
    err_g = torch.mean((pred_g - y_test)**2).item()
    
    pred_r, trace = radon(x_test, use_system2=True)
    err_r = torch.mean((pred_r - y_test)**2).item()
    
    print("-" * 40)
    print(f"📊 SCORECARD:\n   Goliath Error: {err_g:.4f}\n   Radon Error  : {err_r:.4f}")
    print(f"   ➤ Robustness Advantage: {((err_g - err_r)/err_g)*100:.1f}%")
    
    # 3. AUDIT & VIZ
    RadonAuditToolkit.run_audit(radon, goliath)
    print("\n🎥 Plotting 'Ghost in the Machine'...")
    RadonVisualizer.plot_thought_process(trace)

if __name__ == "__main__":
    main()