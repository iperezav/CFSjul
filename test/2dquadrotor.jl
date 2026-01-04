using Symbolics
using LinearAlgebra
using ChenFliessSeries

using DifferentialEquations
using Plots

# ---------------------------------------------------------
# 1. Symbolic Lie derivatives (build numeric evaluator)
# ---------------------------------------------------------

@variables x[1:6]
x_vec = x

# we will vary Ntrunc later; this is just an initial choice
Ntrunc = 4

h = x[1]

# parameters
gg     = 9.81   # Gravitational acceleration (m/s^2)
m     = 0.18    # Mass (kg)
Ixx   = 0.00025 # Mass moment of inertia (kg*m^2)
L     = 0.086   # Arm length (m)


g = hcat(
    [x[4], x[5], x[6], 0, -gg, 0],
    [0, 0, 0, 1/m*sin(x[3]), 1/m*cos(x[3]), -L/Ixx],
    [0, 0, 0, 1/m*sin(x[3]), 1/m*cos(x[3]), L/Ixx]
)

x_val = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]   # make it Float64 to avoid promotion issues

# initial evaluator for Ntrunc = 4
f_L = build_lie_evaluator(h, g, x_vec, Ntrunc)
L_eval = f_L(x_val)   # Vector{Float64}, no Symbolics anywhere

# ---------------------------------------------------------
# 2. Iterated integrals
# ---------------------------------------------------------
dt = 0.001
t = 0:dt:0.1

u0 = one.(t)
u1 = sin.(t)   # elementwise and using broadcasting
u2 = cos.(t)

utemp = vcat(u0', u1', u2')   # 2×T (inputs: u0, u1)

E = iter_int(utemp, dt, Ntrunc)   # purely numeric, from CFSjul

# ---------------------------------------------------------
# 3. Chen–Fliess series (for Ntrunc = 4)
# ---------------------------------------------------------

y_cf = x_val[1] .+ vec(L_eval' * E)   # output h = x2



# ---------------------------------------------------------
# 4. ODE solution using DifferentialEquations.jl
# ---------------------------------------------------------

function twodquad!(dx, x, p, t)
    # input u1(t)
    u1 = sin(t)          # scalar, t is a Float64 here
    u2 = cos(t)

    # same dynamics as symbolic g, but numeric
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = x[6]
    dx[4] = 1/m*sin(x[3])*(u1+u2)
    dx[5] = -gg + (1/m*cos(x[3]))*(u1+u2)
    dx[6] = (L/Ixx)*(u2-u1)
end

x0 = x_val
tspan = (0.0, 1.5)

prob = ODEProblem(twodquad!, x0, tspan)
sol = solve(prob, Tsit5(), saveat = t)

x1_ode = sol[1, :]   # extract x₂(t), since h = x₂


# ---------------------------------------------------------
# 4. Plot both curves
# ---------------------------------------------------------
plot(t, x1_ode,
     label="ODE solution x₁(t)",
     linewidth=3,
     color=:blue)

plot!(t, y_cf,
      label="Chen–Fliess (Ntrunc = 4)",
      linewidth=3,
      linestyle=:dash,
      color=:red)

xlabel!("Time")
ylabel!("y")
title!("ODE vs Chen–Fliess Approximation")
plot!(grid = true)


savefig("Chen-Fliess-series_quadrotor.png")











using DifferentialEquations
using LinearAlgebra
using Plots

# ---------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------
m   = 0.18        # mass (kg)
Ixx = 0.00025     # inertia (kg*m^2)
L   = 0.086       # arm length (m)
g   = 9.81        # gravity (m/s^2)

# ---------------------------------------------------------
# 2. Control inputs (can be time-varying)
# ---------------------------------------------------------
"""
    u = control(t)

Return rotor thrusts (T1, T2) at time t.
Feel free to modify this to test different maneuvers.
"""
function control(t)
    # Example: constant thrusts (hover + torque)
    T1 = sin(t)
    T2 = cos(t)
    return T1, T2
end

# ---------------------------------------------------------
# 3. Planar quadrotor dynamics
# ---------------------------------------------------------
"""
    twodquad!(dx, x, p, t)

State x = [px, pz, θ, vx, vz, ω]
"""
function twodquad!(dx, x, p, t)
    px, pz, θ, vx, vz, ω = x
    T1, T2 = control(t)

    # translational accelerations
    ax = (T1 + T2)/m * sin(θ)
    az = (T1 + T2)/m * cos(θ) - g

    # rotational acceleration
    α = (L/Ixx) * (T2 - T1)

    dx[1] = vx
    dx[2] = vz
    dx[3] = ω
    dx[4] = ax
    dx[5] = az
    dx[6] = α
end

# ---------------------------------------------------------
# 4. Initial condition and simulation
# ---------------------------------------------------------
x0 = [0.0,   # px
      0.0,   # pz
      0.0,   # θ
      0.0,   # vx
      0.0,   # vz
      0.0]   # ω

tspan = (0.0, 2.0)

prob = ODEProblem(twodquad!, x0, tspan)
sol = solve(prob, Tsit5(), saveat=0.001)

# ---------------------------------------------------------
# 5. Plot results
# ---------------------------------------------------------
t = sol.t
px = sol[1, :]
pz = sol[2, :]
θ  = sol[3, :]
vx = sol[4, :]
vz = sol[5, :]
ω  = sol[6, :]

plot(
    plot(t, px, label="pₓ(t)", lw=2),
    plot(t, pz, label="p_z(t)", lw=2),
    plot(t, θ,  label="θ(t)", lw=2),
    plot(t, vx, label="vₓ(t)", lw=2),
    plot(t, vz, label="v_z(t)", lw=2),
    plot(t, ω,  label="ω(t)", lw=2),
    layout=(3,2), size=(900,700),
    title="Planar Quadrotor Trajectory"
)