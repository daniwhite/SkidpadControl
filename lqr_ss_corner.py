""""Set up dynamics for cornering and visualize them."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pydrake.symbolic as sym

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput, SymbolicVectorSystem
from pydrake.all import LinearQuadraticRegulator

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

plant_dynamics = [
    (sym.cos(delta)*S_FL*kappa_F - sym.sin(delta)
     * S_FC*alpha_F + S_RL*kappa_R)/m - v_y*r,
    (sym.sin(delta)*S_FL*kappa_F + sym.cos(delta)
     * S_FC*alpha_F + S_RC*alpha_R)/m + v_x*r,
    l_F*(sym.sin(delta)*S_FL*kappa_F + sym.cos(delta)
         * S_FC*alpha_F) - l_R*S_RC*alpha_R
]

plant_vector_system = SymbolicVectorSystem(
    state=[v_x, v_y, r],
    input=[kappa_F, kappa_R, delta],
    dynamics=plant_dynamics,
    output=[v_x, v_y, r])

position_dynamics = [
    v_y*sym.cos(psi) - v_x*sym.sin(psi),
    v_y*sym.sin(psi) + v_x*sym.cos(psi),
    r
]

position_system = SymbolicVectorSystem(
    state=[x, y, psi],
    input=[v_x, v_y, r],
    dynamics=position_dynamics,
    output=[x, y, psi])

v_bar = 300
delta_bar = 0.01
kappa_F_bar = S_FC*delta_bar/(np.tan(delta_bar)*S_FL)
kappa_R_bar = -np.sin(delta_bar)*delta_bar*(S_FC/S_RL) * \
    (1+1/np.tan(delta_bar)**2)
x_bar = [v_bar, 0, 0]
u_bar = [kappa_F_bar, kappa_R_bar, delta_bar]

# Set up plant and position
builder = DiagramBuilder()
plant = builder.AddSystem(plant_vector_system)
position = builder.AddSystem(position_system)
builder.Connect(plant.get_output_port(0), position.get_input_port(0))

# Set up LQR
lqr_context = plant.CreateDefaultContext()
plant.get_input_port(0).FixValue(lqr_context, u_bar)
lqr_context.SetContinuousState(x_bar)
Q = np.diag([1, 1, 1])
R = np.diag([0.005, 0.005, 0.01])
LQR_Controller = LinearQuadraticRegulator(plant, lqr_context, Q, R)

# Connect plant to LQR
controller = builder.AddSystem(LQR_Controller)
builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

# Set up logging
plant_logger = LogOutput(plant.get_output_port(0), builder)
position_logger = LogOutput(position.get_output_port(0), builder)
diagram = builder.Build()
context = diagram.CreateDefaultContext()

# Create the simulator, and simulate for 10 seconds.
simulator = Simulator(diagram, context)
simulator.Initialize()
simulator.AdvanceTo(0.1)

# Plots
v_x_data = plant_logger.data()[0, :]
v_y_data = plant_logger.data()[1, :]
r_data = plant_logger.data()[2, :]
x_data = position_logger.data()[0, :]
y_data = position_logger.data()[1, :]
psi_data = position_logger.data()[2, :]

vel_mag = (v_x_data**2 + v_y_data**2)**0.5

plt.figure()
plt.subplot(311)
plt.plot(plant_logger.sample_times(), v_x_data)
plt.axhline(x_bar[0], linestyle='--', color="gray")
plt.xlabel('$t$')
plt.ylabel('$v_x$(t)')

plt.subplot(312)
plt.plot(plant_logger.sample_times(), v_y_data)
plt.axhline(x_bar[1], linestyle='--', color="gray")
plt.xlabel('$t$')
plt.ylabel('$v_y$(t)')

plt.subplot(313)
plt.plot(plant_logger.sample_times(), r_data)
plt.axhline(x_bar[2], linestyle='--', color="gray")
plt.xlabel('$t$')
plt.ylabel('$r(t)$')

plt.figure()
plt.subplot(311)
plt.plot(position_logger.sample_times(), x_data)
plt.xlabel('$t$')
plt.ylabel('$x$(t)')

plt.subplot(312)
plt.plot(position_logger.sample_times(), y_data)
plt.xlabel('$t$')
plt.ylabel('$y$(t)')

plt.subplot(313)
plt.plot(position_logger.sample_times(), psi_data)
plt.xlabel('$t$')
plt.ylabel('$\\psi$')

plt.figure()
# psi=0 should point up, psi=pi/2 should point right
plt.polar(np.pi/2-psi_data, position_logger.sample_times())
plt.title("$90\\degree-\\psi$")

plt.xlabel("$t$")
plt.ylabel("$F_Y$")

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

fig = plt.figure()
ax = fig.add_subplot(111, xlim=(min_dim, max_dim), ylim=(min_dim, max_dim))
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.set_aspect('equal')

line, = ax.plot([], [], 'o-')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

dt = position_logger.sample_times()[7] - position_logger.sample_times()[6]


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [x_data[i]]
    thisy = [y_data[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(1, len(y_data)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()
