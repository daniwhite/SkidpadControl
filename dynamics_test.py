""""Set up dynamics for cornering and visualize them."""

import numpy as np
import matplotlib.pyplot as plt
import pydrake.symbolic as sym

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput, SymbolicVectorSystem

# Set up car parameters
m = 276  # kg
Iz = 180.49  # kg*m^2
# Changed to be neutral steer
RWB = 0.5  # 0.583
l_F = RWB*60*0.0254  # m
l_R = (1-RWB)*60*0.0254  # m

# Hand-wavy tire parameters
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
x = sym.Variable("x4")
y = sym.Variable("x5")
psi = sym.Variable("x6")

# Slip angles
alpha_F = sym.atan2(v_y+l_F*r, v_x) - delta
alpha_R = sym.atan2(v_y+l_F*r, v_x)

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
    input=[kappa_F, kappa_R, delta],
    dynamics=dynamics,
    output=[v_x, v_y, r, x, y, psi])

builder = DiagramBuilder()
system = builder.AddSystem(vector_system)
logger = LogOutput(system.get_output_port(0), builder)
builder.ExportInput(system.get_input_port(0))
diagram = builder.Build()

# Initial conditions
x0 = [1, 1, 0, 0, 0, 0]
context = diagram.CreateDefaultContext()
context.SetContinuousState(x0)

# Fix input
u = [1, 1, 0]
inp = system.get_input_port(0)
inp.FixValue(context, u)

# Create the simulator, and simulate for 10 seconds.
simulator = Simulator(diagram, context)
simulator.Initialize()
simulator.AdvanceTo(10)

# Plots
v_x_data = logger.data()[0, :]
v_y_data = logger.data()[1, :]
r_data = logger.data()[2, :]
x_data = logger.data()[3, :]
y_data = logger.data()[4, :]
psi_data = logger.data()[5, :]

vel_mag = (v_x_data**2 + v_y_data**2)**0.5

plt.figure()
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
plt.subplot(311)
plt.plot(logger.sample_times(), x_data)
plt.xlabel('$t$')
plt.ylabel('$x$(t)')

plt.subplot(312)
plt.plot(logger.sample_times(), y_data)
plt.xlabel('$t$')
plt.ylabel('$y$(t)')

plt.subplot(313)
plt.plot(logger.sample_times(), psi_data)
plt.xlabel('$t$')
plt.ylabel('$\\psi$')

plt.figure()
# psi=0 should point up, psi=pi/2 should point right
plt.polar(np.pi/2-psi_data, logger.sample_times())
plt.title("$90\degree-\\psi$")

plt.figure()
plt.scatter(x_data, y_data, c=vel_mag, marker=(
    3, 0, -psi_data[-1]*180/np.pi))  # Same transform as polar, except -90 because thats the rotation for the triangel to point to the right
plt.xlabel("$x$")
plt.ylabel("$y$")
cb = plt.colorbar()
cb.set_label("Speed")
left, right = plt.xlim()
bot, top = plt.ylim()
min_dim = min(left, bot)
max_dim = max(top, right)
plt.xlim(min_dim, max_dim)
plt.ylim(min_dim, max_dim)

plt.figure()
plt.title("Lateral tire forces")
plt.plot(logger.sample_times(), np.arctan2(
    v_y_data+l_F*r_data, v_x_data) - u[2], label="$F_\\{Y\\}$")
plt.xlabel("$t$")
plt.ylabel("$F_Y$")

plt.show()
