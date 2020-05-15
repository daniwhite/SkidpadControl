""""Set up dynamics for cornering and visualize them."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import scipy.optimize

import pydrake.symbolic as sym
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput, SymbolicVectorSystem
from pydrake.all import LinearQuadraticRegulator, MakeFiniteHorizonLinearQuadraticRegulator, FiniteHorizonLinearQuadraticRegulatorOptions
from pydrake.all import eq, MathematicalProgram, Solve, Variable
from pydrake.all import DirectCollocation, PiecewisePolynomial, DirectTranscription

# Set up car parameters
m = 276  # kg
Iz = 180.49  # kg*m^2
RWB = 0.583
l_R = RWB*60*0.0254  # m
l_F = (1-RWB)*60*0.0254  # m

# Hand-wavy tire parameters
# Longitudinal stiffness guessed using tire forces ~ 1000 N, slip ratio ~ 0.1 -> 1000/0.1=1e4
S_FL = 1e4
S_RL = 1e4
# Cornering stiffness guessed using tire forces ~ 1000 N, slip angle ~ 1
S_FC = 1e3
S_RC = 1e3

max_kappa = 0.1
max_alpha = 0.1

simulation_time = 5
fh_lqr_time = 0.5


def get_ss_yaw_moment(beta_bar, omega_bar, r_bar):
    """"Helper function that will be used with brentq to find beta_ bar"""
    ret = (l_F + l_R)*S_RC*beta_bar
    ret /= l_F
    ret += m*r_bar*omega_bar**2*np.cos(beta_bar)
    return ret


def get_ss(r_bar, delta_bar, omega_bar):
    """Find u_bar and x_bar"""

    print("      r_bar: {:.2f}\tomega_bar: {}\tdelta_bar: {}".format(
        r_bar, omega_bar, delta_bar))

    # Solve for beta_bar numerically
    beta_bar = scipy.optimize.brentq(
        get_ss_yaw_moment, -np.pi/2, 0, args=(omega_bar, r_bar))

    alpha_R_bar = beta_bar
    alpha_F_bar = beta_bar - delta_bar

    print("alpha_R_bar: {:.6f}\talpha_F_bar: {:.6f}\t beta_bar: {:.6f}".format(
        alpha_R_bar, alpha_F_bar, beta_bar))

    F_x_bar = -m*r_bar*omega_bar**2*np.sin(beta_bar)
    F_y_bar = -m*r_bar*omega_bar**2*np.cos(beta_bar)

    print("F_x_bar: {:.5f}\tF_y_bar: {:.5f}".format(F_x_bar, F_y_bar))

    kappa_F_bar = -(S_RC * alpha_R_bar +
                    S_FC*alpha_F_bar*np.cos(delta_bar)
                    - F_y_bar)/(S_FL*np.sin(delta_bar))
    kappa_R_bar = -(S_FL*kappa_F_bar*np.cos(delta_bar)
                    - S_FC*alpha_F_bar*np.sin(delta_bar)
                    - F_x_bar)/S_RL

    print("kappa_F_bar: {:.2f}\tkappa_R_bar: {:.2f}".format(
        kappa_F_bar, kappa_R_bar))

    x_bar = [r_bar, 0, omega_bar, beta_bar, 0]
    u_bar = [kappa_F_bar, kappa_R_bar, delta_bar]

    print("x_bar: [" + ("{:.5f}, "*len(x_bar)).format(*x_bar)[:-2] + "]")
    print("u_bar: [" + ("{:.5f}, "*len(u_bar)).format(*u_bar)[:-2] + "]")

    return x_bar, u_bar


def get_plant_and_pos():
    # Inputs
    kappa_F = sym.Variable("kappa_F")
    kappa_R = sym.Variable("kappa_R")
    delta = sym.Variable("delta")

    # Outputs
    r = sym.Variable("r")
    r_dot = sym.Variable("r_dot")
    theta = sym.Variable("theta")
    theta_dot = sym.Variable("theta_dot")
    beta = sym.Variable("beta")
    beta_dot = sym.Variable("beta_dot")

    # v = r_dot r_hat + r theta_dot theta_hat
    v_r_hat = r_dot
    v_theta_hat = r*theta_dot
    gamma = sym.atan2(v_r_hat, v_theta_hat)

    # Slip angles
    alpha_F = beta - delta - gamma
    alpha_R = beta - gamma

    F_xf = sym.cos(delta)*S_FL*kappa_F - sym.sin(delta)*S_FC*alpha_F
    F_xr = S_RL*kappa_R
    F_yf = sym.sin(delta)*S_FL*kappa_F + sym.cos(delta)*S_FC*alpha_F
    F_yr = S_RC*alpha_R

    F_x = F_xf + F_xr
    F_y = F_yr + F_yf

    plant_state = np.array([
        r,
        r_dot,
        theta_dot,
        beta,
        beta_dot


    ])

    theta_ddot = (F_x*sym.cos(beta) - F_y*sym.sin(beta)) / \
        (m*r) - 2*r_dot * theta_dot/r
    plant_dynamics = np.array([
        r_dot,
        (F_x*sym.sin(beta) + F_y*sym.cos(beta))/m + r*theta_dot**2,
        theta_ddot,
        beta_dot,
        theta_ddot - (l_F*F_yf-l_R*F_yr)/Iz
    ])

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

    return plant_vector_system, position_system


def simulate(builder, plant, position, regulator, x0, duration):
    # Set up plant and position
    builder.Connect(plant.get_output_port(0), position.get_input_port(0))
    controller = builder.AddSystem(regulator)
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    # Set up logging
    plant_logger = LogOutput(plant.get_output_port(0), builder)
    position_logger = LogOutput(position.get_output_port(0), builder)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    context.SetContinuousState(x0)

    # Create the simulator and simulate
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    print("Simulating...")
    simulator.AdvanceTo(duration)
    print("Simulation complete!")

    # Return data
    assert len(plant_logger.sample_times()) == len(
        position_logger.sample_times())
    for t1, t2 in zip(plant_logger.sample_times(), position_logger.sample_times()):
        assert t1 == t2

    t = plant_logger.sample_times()
    r_data = plant_logger.data()[0, :]
    r_dot_data = plant_logger.data()[1, :]
    theta_dot_data = plant_logger.data()[2, :]
    beta_data = plant_logger.data()[3, :]
    beta_dot_data = plant_logger.data()[4, :]
    theta_data = position_logger.data()[0, :]

    return t, r_data, r_dot_data, theta_dot_data, beta_data, beta_dot_data, theta_data


fh_lqr_plant_vector_system, fh_lqr_position_system = get_plant_and_pos()
fh_lqr_builder = DiagramBuilder()
fh_lqr_plant = fh_lqr_builder.AddSystem(fh_lqr_plant_vector_system)
fh_lqr_position = fh_lqr_builder.AddSystem(fh_lqr_position_system)

# Get equilibrium
x_bar, u_bar = get_ss(20.0, 0.01, 0.01)
r_bar, r_dot_bar, omega_bar, beta_bar, beta_dot_bar = x_bar

# Configure initial conditions
x0 = np.zeros(6)
x0[0] = r_bar-0.1  # r
x0[1] = 0  # r dot
x0[2] = omega_bar  # theta dot
x0[3] = beta_bar  # beta
x0[4] = 0  # beta  dot
x0[5] = 0  # theta


# Set up direction collocation
prog_dt = 0.01  # 10 ms, like how messages are sent on the car
max_tf = fh_lqr_time
N = int(max_tf/prog_dt)

prog_context = fh_lqr_plant.CreateDefaultContext()
prog = DirectCollocation(fh_lqr_plant,
                         prog_context,
                         num_time_samples=N,
                         minimum_timestep=prog_dt,
                         maximum_timestep=prog_dt*2)
prog.AddEqualTimeIntervalsConstraints()
# prog = DirectTranscription(plant, prog_contexodt, N)
for idx, val in enumerate(x0[: -1]):
    prog.SetInitialGuess(prog.initial_state()[idx], x0[idx])

# Set state to all epsilons -- otherwise, solver gets divide by zero error
prog.SetInitialGuessForAllVariables(np.full((prog.num_vars(), 1), 1e-5))
R_ = 0.01  # Cost on input "effort".
u = prog.input()
# prog.AddRunningCost(R_ * u[0]**2)
# prog.AddRunningCost(R_ * u[1]**2)
# prog.AddRunningCost(R_ * u[2]**2)

# Start at initial condition
# (Have to chop last entry because x0 is init state for whole diagram, so includes theta)
prog.AddBoundingBoxConstraint(
    x0[: -1], x0[: -1], prog.initial_state())
# End at equilibrium
prog.AddBoundingBoxConstraint(
    x_bar, x_bar, prog.final_state())
prog.AddBoundingBoxConstraint(u_bar, u_bar, prog.input(0))
prog.AddBoundingBoxConstraint(u_bar, u_bar, prog.input(N-1))
prog.AddFinalCost(prog.time())

print("Solving....")
result = Solve(prog)
print("Solve complete")
assert result.is_success()
print("Solver found solution!")

# Set up finite-horizon LQR
fh_lqr_context = fh_lqr_plant.CreateDefaultContext()
fh_lqr_plant.get_input_port(0).FixValue(fh_lqr_context, u_bar)
fh_lqr_context.SetContinuousState(x_bar)
Q = np.diag([100, 10, 10, 100, 1])
R = np.diag([0.5, 0.5, 0.1])

options = FiniteHorizonLinearQuadraticRegulatorOptions()
options.x0 = prog.ReconstructStateTrajectory(result)
options.u0 = prog.ReconstructInputTrajectory(result)
options.Qf = Q

traj_x_values = options.x0.vector_values(options.x0.get_segment_times())
traj_u_values = options.x0.vector_values(options.u0.get_segment_times())


plt.figure()
plt.subplot(321)
plt.plot(options.x0.get_segment_times(), traj_x_values[0, :])
plt.axhline(r_bar, color='gray', linestyle='--')
plt.xlabel("$t$")
plt.ylabel("$r$")

plt.subplot(322)
plt.plot(options.x0.get_segment_times(), traj_x_values[1, :])
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("$t$")
plt.ylabel(r"$\dot r$")

plt.subplot(324)
plt.plot(options.x0.get_segment_times(), traj_x_values[2, :])
plt.axhline(omega_bar, color='gray', linestyle='--')
plt.xlabel("$t$")
plt.ylabel(r"$\dot\theta$")

plt.subplot(325)
plt.plot(options.x0.get_segment_times(), traj_x_values[3, :])
plt.axhline(beta_bar, color='gray', linestyle='--')
plt.xlabel("$t$")
plt.ylabel(r"$\beta$")

plt.subplot(326)
plt.plot(options.x0.get_segment_times(), traj_x_values[3, :])
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("$t$")
plt.ylabel(r"$\dot\beta$")

fh_lqr = MakeFiniteHorizonLinearQuadraticRegulator(fh_lqr_plant, fh_lqr_context, t0=options.u0.start_time(),
                                                   tf=options.u0.end_time(), Q=Q, R=R,
                                                   options=options)
end_of_traj = options.x0.value(options.u0.end_time())
print("end_of_traj: [" + ("{:.5f}, " *
                          len(end_of_traj)).format(*end_of_traj.flatten())[:-2] + "]")

fh_lqr_t, fh_lqr_r_data, fh_lqr_r_dot_data, fh_lqr_theta_dot_data, fh_lqr_beta_data, \
    fh_lqr_beta_dot_data, fh_lqr_theta_data, = simulate(
        fh_lqr_builder, fh_lqr_plant, fh_lqr_position, fh_lqr, x0, options.u0.end_time())

ss_x0 = [
    fh_lqr_r_data[-1],
    fh_lqr_r_dot_data[-1],
    fh_lqr_theta_dot_data[-1],
    fh_lqr_beta_data[-1],
    fh_lqr_beta_dot_data[-1],
    fh_lqr_theta_data[-1]
]

switch_time = options.u0.end_time()

print("ss_x0: [" + ("{:.5f}, "*len(ss_x0)).format(*ss_x0)[:-2] + "]")
print("x_bar: [" + ("{:.5f}, "*len(x_bar)).format(*x_bar)[:-2] + "]")

# Set up LQR
lqr_plant_vector_system, lqr_position_system = get_plant_and_pos()
lqr_builder = DiagramBuilder()
plant = lqr_builder.AddSystem(lqr_plant_vector_system)
position = lqr_builder.AddSystem(lqr_position_system)

lqr_context = plant.CreateDefaultContext()
plant.get_input_port(0).FixValue(lqr_context, u_bar)
lqr_context.SetContinuousState(x_bar)
Q = np.diag([1, 100, 100, 1, 1])
R = np.diag([0.5, 0.5, 0.1])
LQR_Controller = LinearQuadraticRegulator(plant, lqr_context, Q, R)

lqr_t, lqr_r_data, lqr_r_dot_data, lqr_theta_dot_data, lqr_beta_data, \
    lqr_beta_dot_data, lqr_theta_data, = simulate(
        lqr_builder, lqr_plant_vector_system, lqr_position_system, LQR_Controller, ss_x0, simulation_time - options.u0.end_time())

lqr_t = lqr_t + fh_lqr_t[-1]
t = np.concatenate((fh_lqr_t, lqr_t))
r_data = np.concatenate((fh_lqr_r_data, lqr_r_data))
r_dot_data = np.concatenate((fh_lqr_r_dot_data, lqr_r_dot_data))
theta_dot_data = np.concatenate((fh_lqr_theta_dot_data, lqr_theta_dot_data))
beta_data = np.concatenate((fh_lqr_beta_data, lqr_beta_data))
beta_dot_data = np.concatenate((fh_lqr_beta_dot_data, lqr_beta_dot_data))
theta_data = np.concatenate((fh_lqr_theta_data, lqr_theta_data))


plt.figure()
plt.subplot(321)
plt.plot(t, r_data)
plt.axhline(r_bar, color='gray', linestyle='--')
plt.axvline(switch_time, color='gray', linestyle='--')
# plt.ylim(0.*r_bar, r_bar*1.05)
plt.xlabel("$t$")
plt.ylabel("$r$")

plt.subplot(322)
plt.plot(t, r_dot_data)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(switch_time, color='gray', linestyle='--')
# plt.ylim(-5, 5)
plt.xlabel("$t$")
plt.ylabel(r"$\dot r$")

plt.subplot(323)
plt.plot(t, theta_data)
plt.axvline(switch_time, color='gray', linestyle='--')
plt.xlabel("$t$")
plt.ylabel(r"$\theta$")

plt.subplot(324)
plt.plot(t, theta_dot_data)
plt.axhline(omega_bar, color='gray', linestyle='--')
plt.axvline(switch_time, color='gray', linestyle='--')
# plt.ylim(0, 2*omega_bar)
plt.xlabel("$t$")
plt.ylabel(r"$\dot\theta$")

plt.subplot(325)
plt.plot(t, beta_data)
plt.axhline(beta_bar, color='gray', linestyle='--')
plt.axvline(switch_time, color='gray', linestyle='--')
# plt.ylim(-2, 1)
plt.xlabel("$t$")
plt.ylabel(r"$\beta$")

plt.subplot(326)
plt.plot(t, beta_dot_data)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(switch_time, color='gray', linestyle='--')
# plt.ylim(-5, 5)
plt.xlabel("$t$")
plt.ylabel(r"$\dot\beta$")

x_data = r_data * np.cos(theta_data)
y_data = r_data * np.sin(theta_data)

# v = r_dot r_hat + r theta_dot theta_hat
speed_r_hat = r_dot_data
speed_theta_hat = r_data*theta_dot_data
speed = np.sqrt(speed_r_hat**2 + speed_theta_hat**2)
max_speed = max(speed)

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

# Add random points off screen just for the colorbar
cmap = cm.get_cmap('plasma')
if max_speed > 0:
    coord = -abs(min_dim)*50 - 10e6
    plt.scatter([coord, coord], [coord, coord],
                c=[0, max_speed], cmap=cmap)
    cb = plt.colorbar()
    cb.set_label("Speed")

# Interpolate to a consistent time
dt = 20e-3
time_scaler = 0.1
print("dt:", dt)
even_t = np.arange(t[0], t[-1], dt*time_scaler)
x_data_even_t = np.interp(even_t, t, x_data)
y_data_even_t = np.interp(even_t, t, y_data)
theta_data_even_t = np.interp(even_t, t, theta_data)
beta_data_even_t = np.interp(even_t, t, beta_data)
speed_data_even_t = np.interp(even_t, t, speed)

plt.grid(True)

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
    if max_speed > 0:
        rgba = cmap(speed_data_even_t[i]/max_speed)
    else:
        rgba = cmap(0)

    # uses same formula for psi, except without the pi/2
    marker_angle = this_theta - this_beta
    marker_angle *= 180/np.pi
    # Normally we'd have to subtract pi/2 or 90, to make sure the triangle is horizontal.
    # However, that isn't necessary here, since we didn't add 90 above

    line.set_data(thisx, thisy)
    line.set_color(rgba)
    line.set_marker((3, 0, marker_angle))
    new_time = even_t[i]
    time_text.set_text(time_template % new_time)
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(
    1, len(even_t)), interval=dt, blit=True, init_func=init)

plt.show()
