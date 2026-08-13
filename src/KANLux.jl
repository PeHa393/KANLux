module KANLux

# ── Exports (uncomment as modules are implemented) ──

# B-spline core (Phase 1)
export B_batch, coef2curve, curve2coef, extend_grid

# Symbolic function library (Phase 2)
export SYMBOLIC_LIB

# Data tools (Phase 3)
export create_dataset

# KANLayer (Phase 4)
export KANLayer

# SymbolicKANLayer (Phase 5)
export SymbolicKANLayer, fix_symbolic!

# MultKAN container (Phase 6)
export MultKAN

# Structural operations (Phase 7)
export prune, refine

# ── Includes (uncomment as modules are implemented) ──

include("bspline.jl")            # Phase 1
include("symbolic_lib.jl")       # Phase 2
include("data.jl")               # Phase 3
include("KANLayer.jl")           # Phase 4
include("SymbolicKANLayer.jl")   # Phase 5
include("MultKAN.jl")            # Phase 6
include("ops_internal.jl")       # Phase 7 (internal helpers)
include("ops.jl")                # Phase 7

end # module KANLux
