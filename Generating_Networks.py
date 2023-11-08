import random

import numpy as np
import matplotlib.pyplot as plt
from iznetwork import*


# Generating Modules
class Modules(IzNetwork):
    def __init__(self, N, Dmax, type_of_network ="exc"):
        super().__init__(N, Dmax)
        self.type = type_of_network
        self.scaling_factor = 17
        self.weight = 1
        self.connections_with_in = 1000
        self.connections_to_inhib = np.zeros((N,N))

    def get_network_pars(self):
        if self.type == "exc":
            a, b, c, d = 0.02, 0.2, -65, 8
        elif self.type == "inhib":
            a, b, c, d = 0.02, 0.2, -50, 2

        a_n, b_n, c_n, d_n =[], [], [], []

        for i in range(self.num):
            a_n.append(a)
            b_n.append(b)
            c_n.append(c)
            d_n.append(d)

        self.setParameters(a_n, b_n, c_n, d_n)

    def setCurrentWithBackgroundFiring(self):
        """
        Set the current input to the network with background firing.
        Ensure that the process occurs every 1 ms.
        """
        # Generate Poisson-distributed random numbers with λ = 0.01 for each neuron
        poisson_values = np.random.poisson(0.01, self._N)

        # Check if the Poisson values are greater than 0 and inject extra current (I = 15)
        for neuron_index, poisson_value in enumerate(poisson_values):
            if poisson_value > 0:
                self._I[neuron_index] = 15  # Inject extra current for spontaneous firing
            else:
                self._I[neuron_index] = 0  # No extra current

    def set_Connections_within(self, weight_scheme, weight_range, scaling_factor):

        # Connections Array
        connection_size = (self._N,self._N)
        connected_neurons = np.full(connection_size, False)

        # 1000 random connections
        num_connections_with_in = self.connections_with_in
        indices = np.random.choice(connection_size[0] * connection_size[1], num_connections_with_in, replace=False)
        connected_neurons.flat[indices] = True

        if weight_scheme == "constant":
            weight = weight_range
        elif weight_scheme == "random":
            weight = np.random.uniform(weight_range[0], weight_range[1], size=num_connections_with_in)
        else:
            raise ValueError('Scheme invalid. Should be "constant" or "random"')

        scaled_weight = weight * scaling_factor
        weights = np.zeros((self._N,self._N))
        weights[connected_neurons] = weights[connected_neurons] + scaled_weight
        self.setWeights(weights)

        delays = np.zeros((self._N,self._N))
        delays = delays.astype(int)

        random_integers = np.random.randint(1, self._Dmax+1, size=num_connections_with_in)

        delays[connected_neurons] += random_integers

        self.setDelays(delays)

class Connection():
    def __init__(self, module1=None, module2=None, weights=None, delays=None):
        self.module1
        self.module2
        self.weights
        self.delays

class Community():
    def __init__(self, modules, connections):
        # List of modules present in the community
        self.modules
        self.connections

    def set_connection(self,module1, module2, weight_scheme, weight_range, scaling_factor, num_connections_from, delay):

        # Connections Array
        connection_size = (module1._N,module2._N)
        connected_neurons = np.full(connection_size, False)

        # 1000 random connections
        indices = np.random.choice(connection_size[0] * connection_size[1], num_connections_from, replace=False)
        connected_neurons.flat[indices] = True

        if weight_scheme == "constant":
            weight = weight_range
        elif weight_scheme == "random":
            weight = np.random.uniform(weight_range[0], weight_range[1], size=num_connections_from)
        else:
            raise ValueError('Scheme invalid. Should be "constant" or "random"')

        scaled_weight = weight * scaling_factor
        weights = np.zeros((module1._N,module._N))
        weights[connected_neurons] = weights[connected_neurons] + scaled_weight
        for connection in self.connections:
            if connection.module1 == module1 and connection.module1 == module1:
                connection.weights = weights
        # Find the connection between the modules

        delays = np.zeros((module1._N,module2._N))
        delays = delays.astype(int)

        random_integers = np.random.randint(1, delay+1, size=num_connections_from)

        delays[connected_neurons] += random_integers

        for connection in self.connections:
            if connection.module1 == module1 and connection.module1 == module1:
                connection.delays = delays

    def set_connection_btw_modules(self, projection_pattern,weight_scheme, weight_range, scaling_factor, delay):
        if projection_pattern == "Focal":
            modules_ex = [self.modules.type_of_network == "exc"]
            selected_ex_module = random.choice(modules_ex)

            num_connections_from = 4

            module_inhib = [self.modules.type_of_network == "inhib"]
            self.set_connection(selected_ex_module, module_inhib, weight_scheme, weight_range, scaling_factor, num_connections_from, delay)

        if projection_pattern == "Diffuse":
            modules_ex = [self.modules.type_of_network == "exc"]
            module_inhib = [self.modules.type_of_network == "inhib"]

            num_connections_from = module_inhib._N
            for module in modules_ex:
                self.set_connection(module_inhib, module, weight_scheme, weight_range, scaling_factor, num_connections_from, delay)



Module_ex = Modules(100,1,"exc")
Module_ex.set_Connections_within("random", (0,1), 17)
breakpoint()



"""
--- Computational Neurodynamics (COMP70100) ---

Solutions to Exercise 2, Question 1. This script depends on the IzNetwork
class, that simulates networks of Izhikevich spiking neurons. Read through the
IzNetwork code and make sure you understand it.




### Question 1a
net = IzNetwork(1, 1)
net.setWeights(np.array([[0]]))
net.setDelays(np.array([[1]]))
net.setParameters(np.array([0.02]), np.array([0.2]), np.array([-65]), np.array([8]))

I = np.array([10])
T = 200
V = np.zeros(T)
for t in range(T):
    net.setCurrent(I)
    net.update()
    V[t] = net.getState()[0][0]

plt.plot(V)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()


### Question 1b
net = IzNetwork(1, 1)
net.setWeights(np.array([[0]]))
net.setDelays(np.array([[1]]))
net.setParameters(np.array([0.02]), np.array([0.25]), np.array([-65]), np.array([2]))

I = np.array([10])
T = 200
V = np.zeros(T)
for t in range(T):
    net.setCurrent(I)
    net.update()
    V[t] = net.getState()[0][0]

plt.plot(V)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()


### Question 1c
def run_simulation(neuron_type, I, transient, duration):

    Utility function to set up a network of a given neuron type, run a
    simulation and record the resulting average firing rate.

    Inputs:
    neuron_type  -- str. Can be 'excitatory' or 'inhibitory'

    I            -- np.array. Constant current to feed the network

    transient    -- int. Number of timesteps (ms) to discard at the start

    duration     -- int. Number of timesteps (ms) to simulate while recording

    Returns:
    firing_rate  -- float

    # First, determine the type of neuron and set parameters accordingly
    if neuron_type == 'excitatory':
        params = {'a': np.array([0.02]), 'b': np.array([0.2]), 'c': np.array([-65]), 'd': np.array([8])}
    elif neuron_type == 'inhibitory':
        params = {'a': np.array([0.02]), 'b': np.array([0.25]), 'c': np.array([-65]), 'd': np.array([2])}
    else:
        raise ValueError('Unknown neuron type.')

    # Initialise the network
    net = IzNetwork(1, 1)
    net.setWeights(np.array([[0]]))
    net.setDelays(np.array([[1]]))
    net.setParameters(**params)
    npI = np.array([I])

    # Run the simulation for some transient time, without recording activity
    for t in range(transient):
        net.setCurrent(npI)
        net.update()

    # Now run again, this time recording the number of spikes
    spike_count = 0
    for t in range(duration):
        net.setCurrent(npI)
        net.update()
        if net.getState()[0][0] > 29:
            spike_count += 1

    # Convert the result to a firing rate in Hertz and return
    firing_rate = spike_count*1000/duration

    return firing_rate

T1 = 100
T2 = 400
I_vec = np.arange(0, 20, 1)

ex_f = list(map(lambda I: run_simulation('excitatory', I, transient=T1, duration=T2), I_vec))
in_f = list(map(lambda I: run_simulation('inhibitory', I, transient=T1, duration=T2), I_vec))

plt.plot(I_vec, ex_f, label='Excitatory')
plt.plot(I_vec, in_f, label='Inhibitory')
plt.legend()
plt.xlabel('Current')
plt.ylabel('Firing rate (Hz)')
plt.show()
"""





