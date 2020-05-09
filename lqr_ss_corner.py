""""Set up dynamics for cornering and visualize them."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

import pydrake.symbolic as sym
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput, SymbolicVectorSystem
from pydrake.all import LinearQuadraticRegulator

# GLOBAL OPTIONS
input_type = "fixed"
fixed_input_type = "force"

# Check options
assert input_type in {"lqr", "fixed"}, "Input type invalid"
assert fixed_input_type in {"slip_angle", "force", "standard"}

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
r = sym.Variable("x1")
r_dot = sym.Variable("x2")
theta = sym.Variable("x3")
theta_dot = sym.Variable("x4")
beta = sym.Variable("x5")
beta_dot = sym.Variable("x6")

# Slip angles
if input_type == "fixed" and fixed_input_type == "slip_angle":
    # More inputs
    alpha_F = sym.Variable("u4")
    alpha_R = sym.Variable("u5")
else:
    alpha_F = beta - delta
    alpha_R = beta

if input_type == "fixed" and fixed_input_type == "force":
    F_xf = sym.Variable("u4")
    F_xr = sym.Variable("u5")
    F_yf = sym.Variable("u6")
    F_yr = sym.Variable("u7")
    F_yr = sym.Variable("u7")
else:
    F_xf = sym.cos(delta)*S_FL*kappa_F - sym.sin(delta) * S_FC*alpha_F
    F_xr = S_RL*kappa_R
    F_yf = sym.sin(delta)*S_FL*kappa_F + sym.cos(delta) * S_FC*alpha_F
    F_yr = S_RC*alpha_R

F_x = F_xf + F_xr
F_y = F_yr + F_yf

plant_state = [
    r,
    r_dot,
    theta_dot,
    beta,
    beta_dot
]

theta_ddot = (F_x*sym.cos(beta) - F_y*sym.sin(beta)) / \
    (m*r) - 2*r_dot * theta_dot/r
plant_dynamics = [
    r_dot,
    (F_x*sym.sin(beta) + F_y*sym.cos(beta))/m + r*theta_dot**2,
    theta_ddot,
    beta_dot,
    theta_ddot - (l_F*F_yf-l_R*F_yr)/Iz
]

if input_type == "fixed" and fixed_input_type != "standard":
    if fixed_input_type == "slip_angle":
        plant_input = [kappa_F, kappa_R, alpha_F, alpha_R, delta]
    elif fixed_input_type == "force":
        plant_input = [F_xf, F_xr, F_yf, F_yr]
else:
    plant_input = [kappa_F, kappa_R, delta]

plant_vector_system = SymbolicVectorSystem(
    state=plant_state,
    input=plant_input,
    dynamics=plant_dynamics,
    output=plant_state)

position_state = [theta]
position_dynamics = [theta_dot]

position_system = SymbolicVectorSystem(
    state=position_state,
    input=plant_state,
    dynamics=position_dynamics,
    output=position_state)

# Set up plant and position
builder = DiagramBuilder()
plant = builder.AddSystem(plant_vector_system)
position = builder.AddSystem(position_system)
builder.Connect(plant.get_output_port(0), position.get_input_port(0))

if input_type == "fixed":
    builder.ExportInput(plant.get_input_port(0))
elif input_type == "lqr":
    v_bar = 300
    delta_bar = 0.01
    print("v_bar: {}\tdelta_bar: {}".format(v_bar, delta_bar))

    # In this edge case, it gets a nan when it divdes 0/0. However, since velocity is constant (and
    # since delta bar is zero, we are going straight), the kappa values should both be zero.
    if delta_bar == 0:
        kappa_F_bar = 0
        kappa_R_bar = 0
    else:
        kappa_F_bar = S_FC*delta_bar/(np.tan(delta_bar)*S_FL)
        kappa_R_bar = -np.sin(delta_bar)*delta_bar*(S_FC/S_RL) * \
            (1+1/np.tan(delta_bar)**2)

    print("kappa_F_bar: {}\\kappa_R_bar: {}".format(kappa_F_bar, kappa_R_bar))

    x_bar = [v_bar, 0, 0]
    u_bar = [kappa_F_bar, kappa_R_bar, delta_bar]

    print("x_bar:", x_bar)
    print("u_bar:", u_bar)

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

# Initial conditions
if input_type == "fixed":
    x0 = [0] * len(plant_state) + [0] * len(position_state)
    x0[0] = 1  # r
    x0[1] = 0  # r dot
    x0[2] = 5  # theta dot
    x0[3] = 0  # beta
    x0[4] = 0  # beta  dot
    x0[5] = 0  # theta
elif input_type == "lqr":
    # x0 = x_bar + [0, 0, 0]
    x0 = [-300, 5, 1, 0, 0, 0]
context.SetContinuousState(x0)

if input_type == "fixed":
    # Fix input
    if fixed_input_type == "slip_angle":
        u = [0, 0, 0.1, 0.1, 0]
    elif fixed_input_type == "force":
        u = [10, 10, 10, 10]
    elif fixed_input_type == "standard":
        u = [1, -1, 0]
    print("Fixed u:", u)
    inp = plant.get_input_port(0)
    inp.FixValue(context, u)

# Create the simulator and simulate
simulator = Simulator(diagram, context)
simulator.Initialize()
simulator.AdvanceTo(10)

plant_state = [
    r,
    r_dot,
    theta_dot,
    beta,
    beta_dot
]
# Plots
r_data = plant_logger.data()[0, :]
r_dot_data = plant_logger.data()[1, :]
theta_dot_data = plant_logger.data()[2, :]
beta_data = plant_logger.data()[3, :]
beta_dot_data = plant_logger.data()[4, :]
theta_data = position_logger.data()[0, :]

plt.figure()
plt.subplot(321)
plt.plot(plant_logger.sample_times(), r_data)
plt.xlabel("$t$")
plt.ylabel("$r$")

plt.subplot(322)
plt.plot(plant_logger.sample_times(), r_dot_data)
plt.xlabel("$t$")
plt.ylabel(r"$\dot r$")

plt.subplot(323)
plt.plot(position_logger.sample_times(), theta_data)
plt.xlabel("$t$")
plt.ylabel("$\theta$")

plt.subplot(324)
plt.plot(plant_logger.sample_times(), theta_dot_data)
plt.xlabel("$t$")
plt.ylabel(r"$\dot\theta$")

plt.subplot(325)
plt.plot(position_logger.sample_times(), beta_data)
plt.xlabel("$t$")
plt.ylabel(r"$\beta$")

plt.subplot(326)
plt.plot(plant_logger.sample_times(), beta_dot_data)
plt.xlabel("$t$")
plt.ylabel(r"$\dot\beta$")

# print("Initial state:", (v_x_data[0], v_y_data[0], r_data[0]))

# speed=(v_x_data**2 + v_y_data**2)**0.5
# max_speed=max(speed)

# plt.figure()
# plt.subplot(311)
# plt.plot(plant_logger.sample_times(), v_x_data)
# if input_type == "lqr":
#     plt.axhline(x_bar[0], linestyle = '--', color = "gray")
# plt.xlabel('$t$')
# plt.ylabel('$v_x$(t)')

# plt.subplot(312)
# plt.plot(plant_logger.sample_times(), v_y_data)
# if input_type == "lqr":
#     plt.axhline(x_bar[1], linestyle = '--', color = "gray")
# plt.xlabel('$t$')
# plt.ylabel('$v_y$(t)')

# plt.subplot(313)
# plt.plot(plant_logger.sample_times(), r_data)
# if input_type == "lqr":
#     plt.axhline(x_bar[2], linestyle = '--', color = "gray")
# plt.xlabel('$t$')
# plt.ylabel('$r(t)$')

# plt.figure()
# plt.subplot(311)
# plt.plot(position_logger.sample_times(), x_data)
# plt.xlabel('$t$')
# plt.ylabel('$x$(t)')

# plt.subplot(312)
# plt.plot(position_logger.sample_times(), y_data)
# plt.xlabel('$t$')
# plt.ylabel('$y$(t)')

# plt.subplot(313)
# plt.plot(position_logger.sample_times(), psi_data)
# plt.xlabel('$t$')
# plt.ylabel('$\\psi$')

x_data = r_data * np.cos(theta_data)
y_data = r_data * np.sin(theta_data)

fig = plt.figure()
plt.plot(x_data, y_data, color='gray', linestyle='--')
plt.xlabel("$x$")
plt.ylabel("$y$")
left, right = plt.xlim()
bot, top = plt.ylim()
min_dim = min(left, bot)
max_dim = max(top, right)
plt.xlim(min_dim, max_dim)
plt.ylim(min_dim, max_dim)

# # Add random points off screen just for the colorbar
# cmap=cm.get_cmap('plasma')
# plt.scatter([-min_dim*50, -min_dim*50], [-min_dim*50, -min_dim*50],
#             c = [0, max_speed], cmap = cmap)
# cb=plt.colorbar()
# cb.set_label("Speed")

# Interpolate to a consistent time
dt = 20e-3
time_scaler = 1
print("dt:", dt)
even_t = np.arange(position_logger.sample_times()[
    0], position_logger.sample_times()[-1], dt*time_scaler)
x_data_even_t = np.interp(
    even_t, position_logger.sample_times(), x_data)
y_data_even_t = np.interp(
    even_t, position_logger.sample_times(), y_data)
theta_data_even_t = np.interp(
    even_t, position_logger.sample_times(), theta_data)
beta_data_even_t = np.interp(
    even_t, plant_logger.sample_times(), beta_data)
# psi_data_even_t = np.interp(
#     even_t, position_logger.sample_times(), psi_data)
# speed_data_even_t = np.interp(
#     even_t, plant_logger.sample_times(), speed)
# even_t *= time_scaler
plt.axhline(0, color='black')
plt.axvline(0, color='black')

ax = plt.gca()
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', markeredgecolor='black',
                markeredgewidth=1, markersize=30)
time_template = 'time = %.1fs'
time_text = ax.text(0.7, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [x_data_even_t[i]]
    thisy = [y_data_even_t[i]]
    this_theta = theta_data_even_t[i]
    this_beta = beta_data_even_t[i]
    # rgba = cmap(speed_data_even_t[i]/max_speed)

    # uses same formula for psi, except without the pi/2
    marker_angle = this_theta - this_beta
    marker_angle *= 180/np.pi
    # Normally we'd have to subtract pi/2 or 90, to make sure the triangle is horizontal.
    # However, that isn't necessary here, since we didn't add 90 above

    line.set_data(thisx, thisy)
    # line.set_color(rgba)
    line.set_marker((3, 0, marker_angle))
    new_time = even_t[i]
    time_text.set_text(time_template % new_time)
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(
    1, len(even_t)), interval=dt, blit=True, init_func=init)

plt.show()
