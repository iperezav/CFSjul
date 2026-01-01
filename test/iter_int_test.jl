
Ntrunc = 3
dt = 0.01
ctrEtemp = Ref(0.0)
t = range(0, stop=1, length=10)
u1 = sin.(t)
u2 = cos.(t)



utemp = stack([u1, u2], dims = 1)

num_input = size(utemp, 1)


total_iterint = num_input*(1-num_input^Ntrunc)/(1-num_input)

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







