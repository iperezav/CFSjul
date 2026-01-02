using Symbolics
using LinearAlgebra
using CFSjul
using DifferentialEquations
using Plots

# ---------------------------------------------------------
# 1. Symbolic Lie derivatives
# ---------------------------------------------------------
@variables x[1:2]
x_vec = x

Ntrunc = 3
h = x[1]

g = hcat([-x[1]*x[2], x[1]*x[2]],
         [x[1],       0],
         [0,         -x[2]])

Ltemp = iter_lie(h, g, x_vec, Ntrunc)

x_val = [1/3, 2/3]
subs = Dict(x[1] => x_val[1], x[2] => x_val[2])

L_eval_num = Symbolics.value.(substitute.(Ltemp, Ref(subs)))
L_eval = Float64.(L_eval_num)   # column vector

# ---------------------------------------------------------
# 2. Iterated integrals (must match Python structure)
# ---------------------------------------------------------
dt = 0.001
t = 0:dt:3

u0 = one.(t)
u1 = sin.(t)
u2 = cos.(t)

utemp = vcat(u0', u1', u2')   # 3×T

E = iter_int(utemp, dt, Ntrunc)

# Chen–Fliess truncated output
y_cf = x_val[1] .+ vec(L_eval' * E)

# ---------------------------------------------------------
# 3. ODE solution using DifferentialEquations.jl
# ---------------------------------------------------------
function lotka_volterra!(dx, x, p, t)
    u1 = sin(t)
    u2 = cos(t)
    dx[1] = -x[1]*x[2] + x[1]*u1
    dx[2] =  x[1]*x[2] - x[2]*u2
end

x0 = x_val
tspan = (0.0, 3.0)

prob = ODEProblem(lotka_volterra!, x0, tspan)
sol = solve(prob, Tsit5(), saveat=t)   # sample at same grid as CF series

x1_ode = sol[1, :]   # extract x₁(t)

# ---------------------------------------------------------
# 4. Plot both curves
# ---------------------------------------------------------
plot(t, x1_ode,
     label="ODE solution x₁(t)",
     linewidth=3,
     color=:blue)

plot!(t, y_cf,
      label="Chen–Fliess (Ntrunc = 3)",
      linewidth=3,
      linestyle=:dash,
      color=:red)

xlabel!("Time")
ylabel!("Value")
title!("ODE vs Chen–Fliess Approximation")
grid!(true)