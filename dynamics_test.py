""""Set up dynamics for cornering and visualize them."""

import math
import matplotlib.pyplot as plt
import pydrake.symbolic as sym

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput, SymbolicVectorSystem

# Set up car parameters (copied from a car parameter file in the sim repo)
m = 276  # kg
Iz = 180.49  # kg*m^2
l_F = 0.583*60*0.0254  # m
l_R = (1-0.583)*60*0.0254  # m

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
delta = sym.Variable("u3")

# Outputs
v_x = sym.Variable("x1")
v_y = sym.Variable("x2")
r = sym.Variable("x3")

# Alphas
alpha_F = sym.atan2(v_y+l_F*r, v_x) - delta
alpha_R = sym.atan2(v_y+l_F*r, v_x)

vector_system = SymbolicVectorSystem(
    state=[v_x, v_y, r],
    input=[kappa_F, kappa_R, delta],
    dynamics=[
        (sym.cos(delta)*S_FL*kappa_F - sym.sin(delta)*S_FC*alpha_F + S_RL*kappa_R)/m,
        (sym.sin(delta)*S_FL*kappa_F + sym.cos(delta)*S_FC*alpha_F + S_RC*alpha_R)/m,
        l_F*(sym.sin(delta)*S_FL*kappa_F + sym.cos(delta)
             * S_FC*alpha_F) - l_R*S_RC*alpha_R
    ],
    output=[v_x, v_y, r])
builder = DiagramBuilder()
system = builder.AddSystem(vector_system)
logger = LogOutput(system.get_output_port(0), builder)
builder.ExportInput(system.get_input_port(0))

diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()
context.SetContinuousState([0, 0, 0])

inp = system.get_input_port(0)
inp.FixValue(context, [0, 0, 0])

# Create the simulator, and simulate for 10 seconds.
simulator = Simulator(diagram, context)
simulator.Initialize()
simulator.AdvanceTo(10)

# Plot the results.
v_x_data = logger.data()[0, :]
v_y_data = logger.data()[1, :]
r_data = logger.data()[2, :]

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

plt.show()
