<div align = "center">
<img src="https://raw.githubusercontent.com/iperezav/CFSjul/main/assets/CFSjul_logo.png" alt="CFSjul" />

---

</div>

# CFSjul

CFSjul is a Julia library to simulate the output of a control system by means of the Chen-Fliess series.

It provides:

- The list of iterated integrals indexed by words of a certain length or less
- The list of Lie derivatives indexed by words of a certain length or less
- A single iterated integral indexed by a given word
- A single Lie derivative indexed by a given word
- The Chen-Fliess series of a nonlinear system


## Overview

CFSjul is a Julia library that contains the following functions:

| Function | Description |
| ---- | --- |
| [**iter_int**](https://github.com/iperezav/CFSjul/blob/main/src/CFSjul.jl) | A function for the numerical computation of a list of iterated integrals |
| [**iter_lie**](https://github.com/iperezav/CFSjul/blob/main/src/CFSjul.jl) | A function for the analytical computation of a list of Lie derivatives |
| [**single_iter_int**](https://github.com/iperezav/CFSjul/blob/main/src/CFSjul.jl) | A function for the numerical computation of a single iterated integral |
| [**single_iter_lie**](https://github.com/iperezav/CFSjul/blob/main/src/CFSjul.jl) | A function for the analytical computation of a single Lie derivative |
| [**chen_fliess_output**](https://github.com/iperezav/CFSjul/blob/main/src/CFSjul.jl) | A function for the analytical computation of a single Lie derivative |

CFSjul is used for:

- Computation of iterated integrals
- Computation of Lie derivatives
- Simulation of the output of a control systems
- Reachability analysis of a control system


# Installation 
Currently, `CFSjul` is guaranteed to run on releases of Julia 1.12.3 onwards.
To install the current release:

```shell
julia> ]
pkg> add CFSjul
```


# Getting Started

## Minimal Example
```julia
# Libraries
using Symbolics
using LinearAlgebra
using CFSjul

# ---------------------------------------------------------
# 1. Lie derivatives
# ---------------------------------------------------------

# Symbolic variable representing the state of the system
@variables x[1:6]
x_vec = x

# Truncation length of words
Ntrunc = 4

# Output of the system
h = x[1] 

# Parameters of the nonlinear control-affine system 
gg     = 9.81   # Gravitational acceleration (m/s^2)
m     = 0.18    # Mass (kg)
Ixx   = 0.00025 # Mass moment of inertia (kg*m^2)
L     = 0.086   # Arm length (m)

# Vector fields
g = hcat(
    [x[4], x[5], x[6], 0, -gg, 0],
    [0, 0, 0, 1/m*sin(x[3]), 1/m*cos(x[3]), -L/Ixx],
    [0, 0, 0, 1/m*sin(x[3]), 1/m*cos(x[3]), L/Ixx]
)

# Initial value
x_val = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0] 

# initial evaluator
f_L = build_lie_evaluator(h, g, x_vec, Ntrunc)

# Lie derivatives or coefficients of the Chen-Fliess series
L_eval = f_L(x_val)   

# ---------------------------------------------------------
# 2. Iterated integrals
# ---------------------------------------------------------

# Time step
dt = 0.001  

# Time interval
t = 0:dt:0.1  

# Inputs
u0 = one.(t)
u1 = sin.(t)   
u2 = cos.(t)

# Stack of the inputs
utemp = vcat(u0', u1', u2')   

# Iterated integrals
E = iter_int(utemp, dt, Ntrunc)   

# ---------------------------------------------------------
# 3. Chen–Fliess series
# ---------------------------------------------------------

y_cf = x_val[1] .+ vec(L_eval' * E)   # output h = x2
```

We can compare this result with a numerical ODE solver

```julia
# Libraries
using DifferentialEquations
using Plots

# ---------------------------------------------------------
# 4. ODE solution using DifferentialEquations.jl
# ---------------------------------------------------------

function twodquad!(dx, x, p, t)
    # Inputs
    u1 = sin(t)          
    u2 = cos(t)

    # Numeric dynamics
    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = x[6]
    dx[4] = 1/m*sin(x[3])*(u1+u2)
    dx[5] = -gg + (1/m*cos(x[3]))*(u1+u2)
    dx[6] = (L/Ixx)*(u2-u1)
end

x0 = x_val
tspan = (0.0, 0.1)

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
      label="Chen–Fliess (Ntrunc = 3)",
      linewidth=3,
      linestyle=:dash,
      color=:red)

xlabel!("Time")
ylabel!("Value")
title!("ODE vs Chen–Fliess Approximation")
plot!(grid = true)
```


<img src="https://raw.githubusercontent.com/iperezav/CFSjul/main/assets/Chen-Fliess-series_quadrotor.png" alt="Chen-Fliess, Ivan Perez Avellaneda" />

For more examples, see the [CFSjul demos](https://github.com/iperezav/CFSjul/tree/main/examples/)


# Resources

- [**PyPi**](https://pypi.org/project/CFSpy/)
- [**Documentation**](https://cfsjul-docs.readthedocs.io/en/latest/)
- [**Issue tracking**](https://github.com/iperezav/CFSjul/issues)


# Contributing

All feedback is welcome. 


# Asking for help
Please reach out if you have any questions:
1. [Github CFSjul discussions](https://github.com/iperezav/CFSjul/discussions/).
2. [Github CFSjul issues](https://github.com/iperezav/CFSjul/issues).


# License

CFSjul is open-source and released under the [MIT License](LICENSE).


# BibTeX
Feel free to cite my work:

```bibtex
@article{iperezave,
  title={CFSjul},
  author={Perez Avellaneda, Ivan},
  journal={GitHub. Note: https://github.com/iperezav/CFSjul},
  volume={1},
  year={2024}
}
```

[issues]: https://github.com/iperezav/CFSjul/issues
[demos]: https://github.com/iperezav/CFSjul/blob/main/examples/

[downloads]: https://pepy.tech/projects/cfspy
[py-versions]: https://pypi.org/project/cfspy/
[pypi-latest-version]: https://pypi.org/project/cfspy/
[license]: https://github.com/iperezav/CFSpy/blob/main/LICENSE