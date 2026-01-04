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

Ntrunc = 4
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
# 4. Error
# ---------------------------------------------------------

Nmax = 6
errors = zeros(Nmax)

for N in 1:Nmax
    y_cf_N = chen_fliess_output(N, x0, g, h, x_vec, dt, utemp)
    errors[N] = maximum(abs.(x1_ode .- y_cf_N))
end


# ---------------------------------------------------------
# 5. Relative Error
# ---------------------------------------------------------

eps_rel = 1e-12

function rel_error(x_true::AbstractVector, x_approx::AbstractVector; eps = 1e-12)
    @assert length(x_true) == length(x_approx)
    denom = abs.(x_true) .+ eps
    return abs.(x_true .- x_approx) ./ denom
end

# example for a single Ntrunc
y_cf = chen_fliess_output(Ntrunc, x0, g, h, x_vec, dt, utemp)
err_rel = rel_error(x1_ode, y_cf; eps = eps_rel)

using Plots
plot(t, err_rel,
     yscale = :log10,
     xlabel = "Time",
     ylabel = "Relative error",
     label = "rel |x₁ - CF|",
     linewidth = 2)



# ---------------------------------------------------------
# 6.  L^2 error over time instead of max error
# ---------------------------------------------------------

function l2_error(x_true::AbstractVector, x_approx::AbstractVector, dt::Real)
    e = x_true .- x_approx
    return sqrt(sum(abs2, e) * dt)
end


Nmax = 6
err_Linf = zeros(Nmax)  # max error
err_L2   = zeros(Nmax)  # L2 error

for N in 1:Nmax
    y_cf_N = chen_fliess_output(N, x0, g, h, x_vec, dt, utemp)
    e = x1_ode .- y_cf_N
    err_Linf[N] = maximum(abs.(e))
    err_L2[N]   = l2_error(x1_ode, y_cf_N, dt)
end

plot(1:Nmax, err_Linf,
     marker = :circle,
     yscale = :log10,
     xlabel = "Truncation depth N",
     ylabel = "Error",
     label = "L∞ error",
     linewidth = 2)

plot!(1:Nmax, err_L2,
      marker = :square,
      yscale = :log10,
      label = "L² error",
      linewidth = 2)



# ---------------------------------------------------------
# 7.  Benchmark runtime vs Ntrunc
# ---------------------------------------------------------

function time_cf(Ntrunc, x0, g, h, x_vec, t, dt)
    t_start = time()
    _ = chen_fliess_output(Ntrunc, x0, g, h, x_vec, dt, utemp)
    return time() - t_start
end

Nmax = 6
runtimes = zeros(Nmax)

for N in 1:Nmax
    runtimes[N] = time_cf(N, x_val, g, h, x_vec, t, dt)
end

plot(1:Nmax, runtimes,
     marker = :circle,
     xlabel = "Truncation depth N",
     ylabel = "Runtime (s)",
     title = "CF Evaluation Runtime vs N",
     linewidth = 2)



# ---------------------------------------------------------
# 8.  3D surface: error vs time vs Ntrunc
# ---------------------------------------------------------

Nmax = 6
Tlen = length(t)

Err_abs = zeros(Nmax, Tlen)  # or Err_rel if you prefer

for (k, N) in enumerate(1:Nmax)
    y_cf_N = chen_fliess_output(N, x_val, g, h, x_vec, dt, utemp)
    Err_abs[k, :] = abs.(x1_ode .- y_cf_N)
end

Ns = collect(1:Nmax)         # N axis
ts = collect(t)              # time axis

surface(ts, Ns, Err_abs,
        xlabel = "Time",
        ylabel = "Truncation depth N",
        zlabel = "Absolute error",
        title = "Error surface |x₁(t) - CF_N(t)|",
        zscale = :log10)




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
plot!(grid = true)


# Error curve
err = abs.(x1_ode .- y_cf)

plot(t, err,
     label = "Error |x₁(t) - CF(t)|",
     xlabel = "Time",
     ylabel = "Absolute Error",
     title = "Chen–Fliess Approximation Error",
     linewidth = 3,
     color = :black)



plot(1:Nmax, errors,
     marker = :circle,
     xlabel = "Truncation depth N",
     ylabel = "Max error over time",
     title = "CF Truncation Error vs N",
     yscale = :log10,
     linewidth = 3)