using Symbolics
using LinearAlgebra
using ChenFliessSeries
using DifferentialEquations
using Plots

# ---------------------------------------------------------
# 1. Symbolic Lie derivatives (build numeric evaluator)
# ---------------------------------------------------------

@variables x[1:2]
x_vec = x

# we will vary Ntrunc later; this is just an initial choice
Ntrunc = 4

h = x[2]

phi(x2) = exp(x2 / (1 + x2))

g = hcat(
    [-x[1] + (1 - x[1]) * phi(x[2]),
     -2x[2] + (1 - x[1]) * phi(x[2])],
    [0, 1]
)

x_val = [0.0, 0.0]   # make it Float64 to avoid promotion issues

# initial evaluator for Ntrunc = 4
f_L = build_lie_evaluator(h, g, x_vec, Ntrunc)
L_eval = f_L(x_val)   # Vector{Float64}, no Symbolics anywhere

# ---------------------------------------------------------
# 2. Iterated integrals
# ---------------------------------------------------------
dt = 0.001
t = 0:dt:1.5

u0 = one.(t)
u1 = -(1 .+ exp.(-2 .* t)) ./ 2   # elementwise and using broadcasting

utemp = vcat(u0', u1')   # 2×T (inputs: u0, u1)

E = iter_int(utemp, dt, Ntrunc)   # purely numeric, from CFSjul

# ---------------------------------------------------------
# 3. Chen–Fliess series (for Ntrunc = 4)
# ---------------------------------------------------------

y_cf = x_val[2] .+ vec(L_eval' * E)   # output h = x2

# ---------------------------------------------------------
# 4. ODE solution using DifferentialEquations.jl
# ---------------------------------------------------------

function CSTR!(dx, x, p, t)
    # input u1(t)
    u1 = -(1 + exp(-2 * t)) / 2          # scalar, t is a Float64 here

    # same dynamics as symbolic g, but numeric
    dx[1] = -x[1] + (1 - x[1]) * exp(x[2] / (1 + x[2]))
    dx[2] = -2 * x[2] + (1 - x[1]) * exp(x[2] / (1 + x[2])) + u1
end

x0 = x_val
tspan = (0.0, 1.5)

prob = ODEProblem(CSTR!, x0, tspan)
sol = solve(prob, Tsit5(), saveat = t)

x2_ode = sol[2, :]   # extract x₂(t), since h = x₂

# ---------------------------------------------------------
# 5. Runtime vs Ntrunc benchmark
# ---------------------------------------------------------

function time_cf(Ntrunc, x0, g, h, x_vec, t, dt)
    t_start = time()

    # (re)build Lie evaluator for this N
    f_LN = build_lie_evaluator(h, g, x_vec, Ntrunc)
    L_evalN = f_LN(x0)

    # recompute iterated integrals for this N
    u0 = one.(t)
    u1 = -(1 .+ exp.(-2 .* t)) ./ 2
    utemp = vcat(u0', u1')
    E_N = iter_int(utemp, dt, Ntrunc)

    _ = x0[2] .+ vec(L_evalN' * E_N)

    return time() - t_start
end

Nmax = 6
runtimes = zeros(Nmax)

for N in 1:Nmax
    runtimes[N] = time_cf(N, x_val, g, h, x_vec, t, dt)
end

p = plot(1:Nmax, runtimes,
         marker = :circle,
         xlabel = "Truncation depth N",
         ylabel = "Runtime (s)",
         title = "CF Evaluation Runtime vs N",
         linewidth = 2)

offset = maximum(runtimes) * 0.03   # 3% vertical offset

for i in 1:Nmax
    annotate!(
        i,
        runtimes[i] + offset,       # move label ABOVE the point
        text(string(round(runtimes[i], digits=4)),
             :black,
             8,
             :center)               # horizontal centering is fine
    )
end


p