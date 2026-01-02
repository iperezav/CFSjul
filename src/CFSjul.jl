module CFSjul

using Symbolics   
using LinearAlgebra



export iter_int, iter_lie

function iter_int(utemp, dt, Ntrunc)

    num_input = size(utemp, 1)



    # Initializes the total number of iterated integrals.
    total_iterint = 0
    # The total number of iterated integrals of word length less than or equal to the truncation length is computed.
    # total_iterint = num_input + num_input**2 + ... + num_input**Ntrunc
    if num_input == 1
        for i in 0:Ntrunc-1
            total_iterint += num_input^(i+1)
        end
    else
            total_iterint = num_input*(1-num_input^Ntrunc)/(1-num_input)
    end
    
    # This is transformed into an integer.
    total_iterint = Int(total_iterint)

    Etemp = zeros(total_iterint, size(utemp, 2))

    ctrEtemp = zeros(Ntrunc+1)

    for i in 0:Ntrunc-1
        ctrEtemp[i+2] =  ctrEtemp[i+1] + num_input^(i+1)
    end

    #ctrEtemp[1] = 1

    sum_acc = cumsum(utemp, dims=2)*dt

    #Etemp[1:num_input,begin:end] = hcat(zeros(num_input, 1), sum_acc[:,begin:end-1])
    Etemp[1:num_input,:] = sum_acc

    for i in 1:Ntrunc-1
        
        start_prev_block = Int(ctrEtemp[i])
        end_prev_block = Int(ctrEtemp[i+1])
        end_current_block = Int(ctrEtemp[i+2])
        num_prev_block = end_prev_block - start_prev_block
        num_current_block = end_current_block - end_prev_block
        #print("i: ", i, " start_prev_block: ", start_prev_block, " end_prev_block: ", end_prev_block, " end_current_block: ", end_current_block, " num_prev_block: ", num_prev_block, " num_current_block: ", num_current_block)
        U_block = repeat(utemp, inner=(num_prev_block,1)) # inputs for current permutation
        #print("U_block size: ", size(U_block))
        prev_int_block = repeat(Etemp[start_prev_block+1:end_prev_block,:], outer=(num_input,1)) # block of previous permutations
        #print("prev_int_block size: ", size(prev_int_block))
        current_int_block = cumsum(U_block.*prev_int_block, dims = 2)*dt
        #print(size(current_int_block))
        #Etemp[end_prev_block+1:end_current_block,:] = hcat(zeros(num_current_block,1), current_int_block[:,begin:end-1])
        Etemp[end_prev_block+1:end_current_block,:] = current_int_block

    end

    return Etemp

end



function iter_lie(h, vector_field, z, Ntrunc)
    @assert Ntrunc ≥ 1 "Ntrunc must be ≥ 1"

    # number of vector fields (columns of g)
    num_vfield = size(vector_field, 2)

    # Initializes the total number of Lie derivatives.
    total_lderiv = 0
    # The total number of Lie derivatives of word length less than or equal to the truncation length is computed.
    # total_lderiv = num_input + num_input**2 + ... + num_input**Ntrunc
    if num_vfield == 1
        for i in 0:Ntrunc-1
            total_lderiv += num_vfield^(i+1)
        end
    else
        # total_lderiv = num_vfield + num_vfield^2 + ... + num_vfield^Ntrunc
        total_lderiv = Int(num_vfield * (1 - num_vfield^Ntrunc) / (1 - num_vfield))
    end

    # repository of all Lie derivatives (symbolic)
    Ltemp = Matrix{Symbolics.Num}(undef, total_lderiv, 1)

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


end # module CFSjul
