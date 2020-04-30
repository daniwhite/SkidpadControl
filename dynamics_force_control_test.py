""""Set up dynamics for cornering and visualize them, but commanding all tire forces directly."""

import math
import matplotlib.pyplot as plt
import pydrake.symbolic as sym

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput, SymbolicVectorSystem

# Set up car parameters (copied from a car parameter file in the sim repo)
m = 276  # kg
Iz = 180.49  # kg*m^2
RWB = 0.5  # 0.583
l_F = RWB*60*0.0254  # m
l_R = (1-RWB)*60*0.0254  # m

# Tire parameters
# Longitudinal stiffness guessed using tire forces ~ 1000 N, slip ratio ~ 0.1 -> 1000/0.1=1e4
S_FL = 1e4
S_RL = 1e4
# Cornering stiffness guessed using tire forces ~ 1000 N, slip angle ~ 1
S_FC = 1e3
S_RC = 1e3

# Inputs
kappa_F = sym.Variable("u1")
kappa_R = sym.Variable("u2")
alpha_F = sym.Variable("u3")
alpha_R = sym.Variable("u4")
delta = sym.Variable("u5")

# Outputs
v_x = sym.Variable("x1")
v_y = sym.Variable("x2")
r = sym.Variable("x3")
x = sym.Variable("x4")
y = sym.Variable("x5")
psi = sym.Variable("x6")


dynamics = [
    (sym.cos(delta)*S_FL*kappa_F - sym.sin(delta)
     * S_FC*alpha_F + S_RL*kappa_R)/m - v_y*r,
    (sym.sin(delta)*S_FL*kappa_F + sym.cos(delta)
     * S_FC*alpha_F + S_RC*alpha_R)/m + v_x*r,
    l_F*(sym.sin(delta)*S_FL*kappa_F + sym.cos(delta)
         * S_FC*alpha_F) - l_R*S_RC*alpha_R,
    v_y*sym.cos(psi) - v_x*sym.sin(psi),
    v_y*sym.sin(psi) + v_x*sym.cos(psi),
    r
]
vector_system = SymbolicVectorSystem(
    state=[v_x, v_y, r, x, y, psi],
    input=[kappa_F, kappa_R, alpha_F, alpha_R, delta],
    dynamics=dynamics,
    output=[v_x, v_y, r, x, y, psi])
builder = DiagramBuilder()
system = builder.AddSystem(vector_system)
logger = LogOutput(system.get_output_port(0), builder)
builder.ExportInput(system.get_input_port(0))

diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()
context.SetContinuousState([1, 1, 0, 0, 0, 0])

inp = system.get_input_port(0)
inp.FixValue(context, [0, 0, 1, 1, 0])

# Create the simulator, and simulate for 10 seconds.
simulator = Simulator(diagram, context)
simulator.Initialize()
simulator.AdvanceTo(10)

# Plot the results.
v_x_data = logger.data()[0, :]
v_y_data = logger.data()[1, :]
r_data = logger.data()[2, :]
x_data = logger.data()[3, :]
y_data = logger.data()[4, :]

vel_mag = (v_x_data**2 + v_y_data**2)**0.5

plt.subplot(311)
plt.plot(logger.sample_times(), v_x_data)
plt.xlabel('$t$')
plt.ylabel('$v_x$(t)')

plt.subplot(312)
plt.plot(logger.sample_times(), v_y_data)
plt.xlabel('$t$')
plt.ylabel('$v_y$(t)')

plt.subplot(313)
plt.plot(logger.sample_times(), r_data)
plt.xlabel('$t$')
plt.ylabel('$r(t)$')

plt.figure()
plt.scatter(x_data, y_data, c=vel_mag)

plt.show()
