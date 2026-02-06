"""
HES-SCO RADON v1.3: CORE LIBRARY
================================================================================
Architect: Naveen George Sunny
Version:   1.3.3 (Production Core)

USAGE:
    from radon_core import RadonUnifiedAgent
    model = RadonUnifiedAgent()
    prediction, thought_trace = model(input_vector, use_system2=True)
"""

import torch
import torch.nn as nn
import numpy as np

# Auto-Install Geoopt if missing
try:
    import geoopt
except ImportError:
    import subprocess
    print("⚙️ Installing Geoopt (Required for Manifold Physics)...")
    subprocess.check_call(["pip", "install", "geoopt"])
    import geoopt

# Enforce Double Precision for Geometric Stability
torch.set_default_dtype(torch.float64)

# ==============================================================================
# 1. THE GEOMETRIC KERNEL
# ==============================================================================

class RadonManifoldCast(nn.Module):
    """
    The Physical Interface.
    Projects raw input vectors onto the R^32 x H^16 x S^16 Product Manifold.
    Features 'Logarithmic Compression' to stabilize infinite inputs.
    """
    def __init__(self):
        super().__init__()
        self.dim_E = 32
        self.dim_H = 16
        self.dim_S = 16
        
        # Manifold Definitions
        self.manifold_H = geoopt.Lorentz(k=1.0, learnable=False) 
        self.manifold_S = geoopt.Sphere()
        
    def forward(self, x_raw):
        # A. Split Raw Vector
        raw_E, raw_H, raw_S = torch.split(x_raw, [self.dim_E, self.dim_H, self.dim_S], dim=-1)
        
        # B. Euclidean Cast (Identity)
        z_E = raw_E 
        
        # C. Hyperbolic Cast (Log-Compressed)
        # Prevents OOD explosion by compressing infinite variance into finite curvature.
        norm_H = raw_H.norm(p=2, dim=-1, keepdim=True)
        compression = torch.log1p(norm_H) / (norm_H + 1e-6)
        safe_raw_H = raw_H * compression
        
        # ExpMap: Tangent Space -> Manifold Surface
        origin_H = self.manifold_H.origin(raw_H.shape[:-1] + (self.dim_H+1,)).to(x_raw.device)
        zeros = torch.zeros_like(safe_raw_H[..., :1])
        tangent_vector = torch.cat([zeros, safe_raw_H], dim=-1)
        z_H = self.manifold_H.expmap(origin_H, tangent_vector)
        
        # D. Spherical Cast (Projected)
        z_S = self.manifold_S.projx(raw_S)
        
        return z_E, z_H, z_S

# ==============================================================================
# 2. INTRINSIC OPERATORS
# ==============================================================================

class IntrinsicMobiusLinear(nn.Module):
    """
    Manifold-Aware Linear Layer.
    Performs operations in the local Tangent Space to respect curvature.
    """
    def __init__(self, manifold, in_features, out_features):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        origin = self.manifold.origin(x.shape).to(x.device)
        x_tan = self.manifold.logmap(origin, x)
        y_raw = self.linear(x_tan)
        # Force Tangency: Enforce the 'Law of Physics' on the NN output
        y_tan = self.manifold.proju(origin, y_raw)
        y = self.manifold.expmap(origin, y_tan)
        return y

class EpiplexityGate(nn.Module):
    """
    The 'Feeling' of the Model.
    Measures geometric tension to trigger System 2 reasoning.
    Output: 0.0 (Calm) to 1.0 (Panic).
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
# 3. SYSTEM 2 ENGINE (THE GHOST)
# ==============================================================================

class RadonSystem2Engine(nn.Module):
    """
    The Cognitive Processor.
    Implements the 'Thinking Loop' using Riemannian Gradient Descent.
    """
    def __init__(self):
        super().__init__()
        self.caster = RadonManifoldCast()
        
        # Intrinsic Processors
        self.proc_H = IntrinsicMobiusLinear(self.caster.manifold_H, 17, 17)
        self.proc_S = IntrinsicMobiusLinear(self.caster.manifold_S, 16, 16)
        self.proc_E = nn.Linear(32, 32)
        
        # Monitor
        self.gate = EpiplexityGate(dim_total=32+17+16)
        
    def forward(self, x_raw, thinking_budget=30, lr=0.05):
        # 1. System 1 (Reflex)
        z_E, z_H, z_S = self.caster(x_raw)
        z_E = self.proc_E(z_E)
        z_H = self.proc_H(z_H)
        z_S = self.proc_S(z_S)
        
        # 2. Detach for Thinking
        z_E_think = z_E.clone().detach().requires_grad_(True)
        z_H_think = z_H.clone().detach().requires_grad_(True)
        z_S_think = z_S.clone().detach().requires_grad_(True)
        
        trace = []
        
        # 3. System 2 Loop (Optimization)
        if thinking_budget > 0:
            for t in range(thinking_budget):
                energy = self.gate(z_E_think, z_H_think, z_S_think)
                
                # Stop if Epiplexity is low enough
                if energy.mean() < 0.1: break
                    
                # Calculate Gradient of "Surprise"
                grad_E, grad_H, grad_S = torch.autograd.grad(
                    energy.sum(), [z_E_think, z_H_think, z_S_think]
                )
                
                # Riemannian Update (Sliding along the curve)
                with torch.no_grad():
                    z_E_think.sub_(lr * grad_E)
                    
                    rgrad_H = self.caster.manifold_H.egrad2rgrad(z_H_think, grad_H)
                    z_H_think.data = self.caster.manifold_H.retr(z_H_think, -lr * rgrad_H)
                    
                    rgrad_S = self.caster.manifold_S.egrad2rgrad(z_S_think, grad_S)
                    z_S_think.data = self.caster.manifold_S.retr(z_S_think, -lr * rgrad_S)
                
                # Reset Gradients
                z_E_think.grad = None
                z_H_think.grad = None
                z_S_think.grad = None
                
                trace.append(energy.mean().item())

        return (z_E_think, z_H_think, z_S_think), trace

# ==============================================================================
# 4. GEOMETRIC MEMORY CORE
# ==============================================================================

class RadonMemoryCore(nn.Module):
    """
    Long-Term Storage.
    - Frame Operator (SPD): Worldview/Covariance.
    - Task Basis (Stiefel): Context Subspaces.
    """
    def __init__(self, dim_H=16):
        super().__init__()
        self.dim_H = dim_H
        
        # SPD Manifold (Covariance)
        self.manifold_SPD = geoopt.SymmetricPositiveDefinite()
        self.frame_operator = geoopt.ManifoldParameter(
            torch.eye(dim_H).unsqueeze(0), manifold=self.manifold_SPD
        )
        
        # Stiefel Manifold (Subspaces)
        self.manifold_St = geoopt.Stiefel()
        q, _ = torch.linalg.qr(torch.randn(1, dim_H, 2))
        self.task_basis = geoopt.ManifoldParameter(q, manifold=self.manifold_St)

    def update_worldview(self, z_batch, lr=0.01):
        """ Continual Learning via Geodesic Averaging """
        z_cen = z_batch - z_batch.mean(0, keepdim=True)
        cov = (z_cen.t() @ z_cen) / (z_batch.shape[0] - 1 + 1e-8)
        cov = cov + 1e-5 * torch.eye(self.dim_H, device=z_batch.device)
        target = cov.unsqueeze(0)
        
        vec = self.manifold_SPD.logmap(self.frame_operator, target)
        self.frame_operator.data = self.manifold_SPD.expmap(self.frame_operator, lr * vec)
        return self.frame_operator

# ==============================================================================
# 5. THE UNIFIED AGENT (API)
# ==============================================================================

class RadonUnifiedAgent(nn.Module):
    """
    The High-Level Wrapper.
    Use this class for your applications.
    """
    def __init__(self):
        super().__init__()
        self.brain = RadonSystem2Engine()
        self.memory = RadonMemoryCore()
        
        # Poincare Readout Layer
        # Guarantees bounded inputs for the final decision
        self.readout = nn.Sequential(
            nn.Linear(32 + 16 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def lorentz_to_poincare(self, z_H):
        """ Isometry: Unbounded Lorentz -> Bounded Poincare Ball """
        t = z_H[..., 0:1]
        x = z_H[..., 1:]
        return x / (1 + t)

    def forward(self, x, use_system2=True):
        # 1. Run Cognitive Loop
        (z_E, z_H, z_S), trace = self.brain(x, thinking_budget=30 if use_system2 else 0)
        
        # 2. Project to Bounded Domain (Safety)
        z_P = self.lorentz_to_poincare(z_H)
        z_E_bounded = torch.tanh(z_E)
        
        # 3. Decision
        state_flat = torch.cat([z_E_bounded, z_P, z_S], dim=-1)
        decision = self.readout(state_flat)
        
        return decision, trace
