"""
generate_reference.py — Generate deterministic PyKAN reference data for KANLux verification.

Run: python generate_reference.py
Output: test/reference/*.npy (≥10 files)

Requires: pykan installed (pip install -e . in repo root)
          numpy (with pykan)

All outputs are deterministic — same seed → same .npy every run.
"""

import os, sys, random

# Ensure pykan is importable (repo root is parent of this script's grandparent)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch

# ── KANs need double precision ──
torch.set_default_dtype(torch.float64)

# ── Fixed seed for full determinism ──
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Output directory ──
OUT_DIR = os.path.join(os.path.dirname(__file__), "reference")
os.makedirs(OUT_DIR, exist_ok=True)


def save(name: str, tensor: torch.Tensor):
    """Save a tensor as .npy to reference/ directory."""
    path = os.path.join(OUT_DIR, f"{name}.npy")
    arr = tensor.detach().cpu().numpy()
    np.save(path, arr)
    print(f"  ✓ {name}.npy  shape={arr.shape}  dtype={arr.dtype}")


# ══════════════════════════════════════════════════════════════════
# 1. B-spline primitives (kan.spline)
# ══════════════════════════════════════════════════════════════════

from kan.spline import B_batch, coef2curve, curve2coef, extend_grid

# Fixed parameters for all spline tests
IN_DIM = 2
OUT_DIM = 3
NUM = 5       # grid intervals
K = 3         # cubic B-spline
BATCH = 100

# ── B_batch ──
x_bb = torch.rand(BATCH, IN_DIM)                              # (batch, in_dim)
grid_bb = torch.linspace(-1, 1, steps=NUM + 1)[None, :].expand(IN_DIM, NUM + 1)
grid_bb = extend_grid(grid_bb, k_extend=K)                     # (in_dim, num+1+2k)
out_bb = B_batch(x_bb, grid_bb, k=K)                           # (batch, in_dim, num+k)

save("b_batch_input_x", x_bb)
save("b_batch_grid", grid_bb)
save("b_batch_output", out_bb)

# ── coef2curve ──
x_c2c = torch.rand(BATCH, IN_DIM)                              # (batch, in_dim)
grid_c2c = grid_bb.clone()                                     # (in_dim, num+1+2k)
coef_c2c = torch.randn(IN_DIM, OUT_DIM, NUM + K)               # (in_dim, out_dim, num+k)
out_c2c = coef2curve(x_c2c, grid_c2c, coef_c2c, k=K)           # (batch, in_dim, out_dim)

save("coef2curve_x_eval", x_c2c)
save("coef2curve_grid", grid_c2c)
save("coef2curve_coef", coef_c2c)
save("coef2curve_output", out_c2c)

# ── curve2coef (round-trip test: coef → curve → coef') ──
x_c2c_inv = x_c2c.clone()                                      # (batch, in_dim)
y_c2c_inv = out_c2c.clone()                                    # (batch, in_dim, out_dim)
coef_inv = curve2coef(x_c2c_inv, y_c2c_inv, grid_c2c, k=K)     # (in_dim, out_dim, num+k)

save("curve2coef_x_eval", x_c2c_inv)
save("curve2coef_y_eval", y_c2c_inv)
save("curve2coef_grid", grid_c2c)
save("curve2coef_output", coef_inv)

# ── extend_grid ──
grid_ext = torch.linspace(-1, 1, steps=NUM + 1)[None, :].expand(IN_DIM, NUM + 1)
grid_ext_out = extend_grid(grid_ext, k_extend=K)                # (in_dim, num+1+2k)

save("extend_grid_input", grid_ext)
save("extend_grid_output", grid_ext_out)


# ══════════════════════════════════════════════════════════════════
# 2. KANLayer (kan.KANLayer)
# ══════════════════════════════════════════════════════════════════

from kan.KANLayer import KANLayer

kl = KANLayer(in_dim=IN_DIM, out_dim=OUT_DIM, num=NUM, k=K,
              noise_scale=0.1, device='cpu')

x_kl = torch.rand(BATCH, IN_DIM)                               # (batch, in_dim)
y_kl, preacts_kl, postacts_kl, postspline_kl = kl.forward(x_kl)

save("kanlayer_input", x_kl)
save("kanlayer_output", y_kl)            # (batch, out_dim)
save("kanlayer_preacts", preacts_kl)     # (batch, out_dim, in_dim)
save("kanlayer_postacts", postacts_kl)   # (batch, out_dim, in_dim)
save("kanlayer_postspline", postspline_kl)  # (batch, out_dim, in_dim)

# Also save KANLayer's internal parameters for the Julia side to initialize from
save("kanlayer_grid", kl.grid)           # (in_dim, num+1+2k)
save("kanlayer_coef", kl.coef)           # (in_dim, out_dim, num+k)
save("kanlayer_mask", kl.mask)           # (in_dim, out_dim)
save("kanlayer_scale_base", kl.scale_base)  # (in_dim, out_dim)
save("kanlayer_scale_sp", kl.scale_sp)      # (in_dim, out_dim)


# ══════════════════════════════════════════════════════════════════
# 3. SymbolicKANLayer (kan.Symbolic_KANLayer)
# ══════════════════════════════════════════════════════════════════

from kan.Symbolic_KANLayer import Symbolic_KANLayer

skl = Symbolic_KANLayer(in_dim=IN_DIM, out_dim=OUT_DIM, device='cpu')

# Fix each edge to a known symbolic function (simple ones for deterministic output)
# Set affine = [1, 0, 1, 0] (identity transform) for all edges
# Use 'sin' for (0,0), 'x^2' for (0,1), 'exp' for (1,0), 'x' for (1,1), 'abs' for (2,0), 'tanh' for (2,1)
fun_map = {
    (0, 0): "sin",   (0, 1): "x^2",
    (1, 0): "exp",   (1, 1): "x",
    (2, 0): "abs",   (2, 1): "tanh",
}
for (j, i), fn in fun_map.items():
    skl.fix_symbolic(i=i, j=j, fun_name=fn, verbose=False)

x_skl = torch.rand(BATCH, IN_DIM)                              # (batch, in_dim)
y_skl, postacts_skl = skl.forward(x_skl, singularity_avoiding=False)

save("symbolic_kanlayer_input", x_skl)
save("symbolic_kanlayer_output", y_skl)         # (batch, out_dim)
save("symbolic_kanlayer_postacts", postacts_skl)  # (batch, out_dim, in_dim)

# Also save mask and affine for Julia-side parameter initialization
save("symbolic_kanlayer_mask", skl.mask)       # (out_dim, in_dim)
save("symbolic_kanlayer_affine", skl.affine)   # (out_dim, in_dim, 4)


# ══════════════════════════════════════════════════════════════════
# 4. MultKAN (kan.MultKAN) — 2-layer end-to-end
# ══════════════════════════════════════════════════════════════════

from kan import KAN

# Two modes: with and without symbolic branch
# Mode A: symbolic_enabled=True (default, full KAN)
model_a = KAN(width=[IN_DIM, 5, 1], grid=NUM, k=K, seed=SEED,
              auto_save=False, device='cpu')
model_a.speed()  # disable symbolic for speed (just want deterministic numerical branch)

x_mk = torch.rand(BATCH, IN_DIM)
y_mk_numerical = model_a(x_mk)

save("multkan_input", x_mk)
save("multkan_output_numerical", y_mk_numerical)  # (batch, 1) — numerical only

# Save MultKAN's internal parameters so the Julia test can reconstruct ps/st and
# compare the end-to-end forward pass without relying on RNG replication.
for l in range(model_a.depth):
    layer = model_a.act_fun[l]
    save(f"multkan_act{l}_grid", layer.grid)
    save(f"multkan_act{l}_coef", layer.coef)
    save(f"multkan_act{l}_mask", layer.mask)
    save(f"multkan_act{l}_scale_base", layer.scale_base)
    save(f"multkan_act{l}_scale_sp", layer.scale_sp)

for l in range(model_a.depth):
    save(f"multkan_node_bias{l}", model_a.node_bias[l])
    save(f"multkan_node_scale{l}", model_a.node_scale[l])
    save(f"multkan_subnode_bias{l}", model_a.subnode_bias[l])
    save(f"multkan_subnode_scale{l}", model_a.subnode_scale[l])

# Mode B: symbolic_enabled=True (full hybrid)
# Reset seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

model_b = KAN(width=[IN_DIM, 5, 1], grid=NUM, k=K, seed=SEED,
              auto_save=False, symbolic_enabled=True, device='cpu')
# Don't call speed() — keep symbolic branch active
# But symbolic functions start as '0' (identity×0 = zero), so symbolic branch contributes 0 anyway
y_mk_full = model_b(x_mk)

save("multkan_output_full", y_mk_full)  # (batch, 1) — with symbolic (currently all zeros)


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

print(f"\nDone! Generated {len(os.listdir(OUT_DIR))} .npy files in {OUT_DIR}/")
print("All outputs are deterministic (seed=42). Re-run gives identical results.")
