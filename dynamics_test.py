""""Set up dynamics for cornering and visualize them."""

import math
import matplotlib.pyplot as plt
import pydrake.symbolic as sym

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogOutput, SymbolicVectorSystem

x = sym.Variable("x")
vector_system = SymbolicVectorSystem(state=[x], dynamics=[0.1], output=[x])

builder = DiagramBuilder()
system = builder.AddSystem(vector_system)
logger = LogOutput(system.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()
context.SetContinuousState([0.9])

# Create the simulator, and simulate for 10 seconds.
simulator = Simulator(diagram, context)
simulator.Initialize()
simulator.AdvanceTo(10)

# Plot the results.
plt.figure()
plt.plot(logger.sample_times(), logger.data().transpose())
plt.xlabel('t')
plt.ylabel('y(t)')
plt.show()
