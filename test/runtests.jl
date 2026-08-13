# KANLux test suite — assembled in Phase 8
# Individual test files are included as modules are implemented.
#
# Usage:
#   cd KANLux && julia --project -e 'using Pkg; Pkg.test()'
#
# Or run individual files:
#   julia --project=KANLux -e 'include("KANLux/test/test_bspline.jl")'

using KANLux
using Test

# Phase 1: B-spline core
include("test_bspline.jl")

# Phase 2: Symbolic function library
include("test_symbolic_lib.jl")

# Phase 3: Data tools
include("test_data.jl")

# Phase 4: KANLayer
include("test_kanlayer.jl")

# Phase 5: SymbolicKANLayer
include("test_symbolic_kanlayer.jl")

# Phase 6: MultKAN
include("test_multkan.jl")

# Phase 7: Structural operations
include("test_ops.jl")

# Phase 8: Global gradient cross-validation
include("test_gradients.jl")
