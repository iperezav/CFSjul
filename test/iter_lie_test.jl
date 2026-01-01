using Symbolics
using LinearAlgebra

# Define symbolic variables
@variables x[1:2]
#x = Symbolics.variables(:x, 1:2)

# Parameters
Ntrunc = 3
h = x[1]

# Define vector field as a 2×3 matrix
g = hcat([-x[1]*x[2], x[1]*x[2]], [x[1], 0], [0, -x[2]])
vector_field = g
num_vfield = size(vector_field, 2)

# Compute total number of Lie derivatives
total_lderiv = Int(num_vfield * (1 - num_vfield^Ntrunc) / (1 - num_vfield))

# Allocate symbolic matrix for Lie derivatives
Ltemp = Matrix{Num}(undef, total_lderiv, 1)

# Track block boundaries
ctrLtemp = zeros(Int, Ntrunc + 1)
for i in 0:Ntrunc-1
    ctrLtemp[i+2] = ctrLtemp[i+1] + num_vfield^(i+1)
end

# First-order Lie derivatives
for i in 1:num_vfield
    Ltemp[i, 1] = (Symbolics.jacobian([h], x) * vector_field[:, i])[1]
end

# Higher-order Lie derivatives
for i in 1:Ntrunc-1
    start_prev_block = ctrLtemp[i] + 1
    end_prev_block = ctrLtemp[i+1]
    end_current_block = ctrLtemp[i+2]
    num_prev_block = end_prev_block - start_prev_block + 1

    # Track where to write new entries
    write_index = end_prev_block + 1

    for k in 1:num_vfield
        for j in 0:num_prev_block-1
            idx = start_prev_block + j
            Ltemp[write_index, 1] = (Symbolics.jacobian([Ltemp[idx, 1]], x) * vector_field[:, k])[1]
            write_index += 1
        end
    end
end