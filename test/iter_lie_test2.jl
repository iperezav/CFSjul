using Symbolics
using LinearAlgebra

function iter_lie(h, vector_field, z, Ntrunc)
    @assert Ntrunc ≥ 1 "Ntrunc must be ≥ 1"

    # number of vector fields (columns of g)
    num_vfield = size(vector_field, 2)

    # total_lderiv = num_vfield + num_vfield^2 + ... + num_vfield^Ntrunc
    total_lderiv = Int(num_vfield * (1 - num_vfield^Ntrunc) / (1 - num_vfield))

    # repository of all Lie derivatives (symbolic)
    Ltemp = Matrix{Num}(undef, total_lderiv, 1)

    # ctrLtemp[k+1] = num_vfield + num_vfield^2 + ... + num_vfield^k
    ctrLtemp = zeros(Int, Ntrunc + 1)  # ctrLtemp[1] = 0
    for i in 0:Ntrunc-1
        ctrLtemp[i+2] = ctrLtemp[i+1] + num_vfield^(i+1)
    end
    # so:
    #   words of length 1 occupy indices (1 : ctrLtemp[2])
    #   words of length 2 occupy (ctrLtemp[2]+1 : ctrLtemp[3])
    #   etc.

    # --- Lie derivatives of words of length 1 ---
    # LT = [h].jacobian(z) * g   (1×n) * (n×m) = 1×m
    LT = Symbolics.jacobian([h], z) * vector_field   # 1×num_vfield

    # Python's LT.reshape(m,1) flattens row-major
    # Row-major flatten in Julia: vec(permutedims(LT))
    LT_flat = vec(permutedims(LT))  # length = num_vfield

    # Fill first block
    Ltemp[1:num_vfield, 1] .= LT_flat

    # --- Higher-order Lie derivatives ---
    for i in 1:Ntrunc-1
        # Python:
        #   start_prev_block = ctrLtemp[i-1]
        #   end_prev_block   = ctrLtemp[i]
        #   end_current_block= ctrLtemp[i+1]
        #
        # Here, ctrLtemp[1] = 0, so:
        #   for i=1 (len=2 words): previous length=1 block is (0 : ctrLtemp[2])
        start_prev_block  = ctrLtemp[i]      # count before previous block
        end_prev_block    = ctrLtemp[i+1]    # cumulative up to previous block
        end_current_block = ctrLtemp[i+2]    # cumulative up to current block

        num_prev_block    = end_prev_block - start_prev_block
        num_current_block = end_current_block - end_prev_block
        @assert num_current_block == num_prev_block * num_vfield

        # LT = Ltemp[start_prev_block:end_prev_block, 0] in Python (0-based, end exclusive)
        # In Julia (1-based, end inclusive):
        LT_prev = Ltemp[start_prev_block+1:end_prev_block, 1]  # size = num_prev_block×1

        # Jacobian wrt z: each entry in LT_prev is a scalar function of z
        # Symbolics.jacobian(LT_prev, z) → (num_prev_block×n)
        # Multiply by g (n×m) → (num_prev_block×m)
        LT_mat = Symbolics.jacobian(LT_prev, z) * vector_field  # num_prev_block×num_vfield

        # Python: LT = LT.reshape(LT.shape[0]*LT.shape[1], 1)
        # i.e., row-major flatten.
        LT_flat = vec(permutedims(LT_mat))  # length = num_prev_block * num_vfield

        # Fill next block
        Ltemp[end_prev_block+1:end_current_block, 1] .= LT_flat
    end

    return Ltemp
end


using Symbolics
using LinearAlgebra

# symbolic variables
@variables x[1:2]
x_vec = x  # for clarity

# parameters
Ntrunc = 4
h = x[1]

# vector field g (2×3)
g = hcat([-x[1]*x[2], x[1]*x[2]],
         [x[1],       0],
         [0,         -x[2]])

# compute Lie derivatives (symbolic)
Ltemp = iter_lie(h, g, x_vec, Ntrunc)

# evaluate at x = [1/3, 2/3]
subs = Dict(x[1] => 1//3, x[2] => 2//3)
L_eval_num = Symbolics.value.(substitute.(Ltemp, Ref(subs)))  # Matrix{Number}
L_eval = Float64.(L_eval_num)                                # Matrix{Float64}

# L_eval should now correspond (up to SymPy/Julia simplification differences)
# to Ceta from your Python code.