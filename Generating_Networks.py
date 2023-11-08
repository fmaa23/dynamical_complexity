import random

import numpy as np
import matplotlib.pyplot as plt
from iznetwork import*


# Generating Modules
class Modules(IzNetwork):
    def __init__(self, N, Dmax, type_of_network ="exc",scaling_factor = 1, weight = 1, connections_with_in = 1000):
        super().__init__(N, Dmax)
        self.type_of_network = type_of_network
        self.scaling_factor = scaling_factor
        self.weight = weight
        self.connections_with_in = connections_with_in

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
        self.module1 = module1
        self.module2 = module2
        self.weights = weights
        self.delays = delays

class Community():
    def __init__(self, modules=[], connections=[]):
        # List of modules present in the community
        self.modules= modules
        self.connections = connections

    def set_connection(self,module1, module2, weight_scheme, weight_range, scaling_factor, num_connections_from, delay):

        # Connections Array
        connection_size = (module1._N,module2._N)
        connected_neurons = np.full(connection_size, False)

        # 1000 random connections
        indices = np.random.choice(connection_size[0], num_connections_from, replace=False)

        connected_neurons[indices,:] = True
        print(connected_neurons)

        if weight_scheme == "constant":
            weight = weight_range
        elif weight_scheme == "random":
            weight = np.random.uniform(weight_range[0], weight_range[1], size=num_connections_from*module2._N)
        else:
            raise ValueError('Scheme invalid. Should be "constant" or "random"')

        scaled_weight = weight * scaling_factor
        weights = np.zeros((module1._N,module2._N))

        weights[connected_neurons] = weights[connected_neurons] + scaled_weight
        connection = Connection()
        connection.module1 = module1
        connection.module2 = module2
        connection.weights = weights
        # Find the connection between the modules

        delays = np.zeros((module1._N,module2._N))
        delays = delays.astype(int)

        random_integers = np.random.randint(1, delay+1, size=num_connections_from*module2._N)

        delays[connected_neurons] += random_integers

        connection.delays = delays
        self.connections.append(connection)


    def set_connection_btw_modules(self, projection_pattern,weight_scheme, weight_range, scaling_factor, delay):
        if projection_pattern == "Focal":
            modules_ex_idx = [self.modules[i].type_of_network == "exc" for i in range(len(self.modules))]
            modules_ex = [self.modules[i] for i, value in enumerate(modules_ex_idx) if value]

            selected_ex_module= random.choice(modules_ex)
            num_connections_from = 4

            modules_inhib_idx = [self.modules[i].type_of_network == "inhib" for i in range(len(self.modules))]
            module_inhib = [self.modules[i] for i, value in enumerate(modules_inhib_idx) if value]
            module_inhib = module_inhib[0]

            self.set_connection(selected_ex_module, module_inhib, weight_scheme, weight_range, scaling_factor, num_connections_from, delay)

        if projection_pattern == "Diffuse":
            modules_ex_idx = [self.modules[i].type_of_network == "exc" for i in range(len(self.modules))]
            modules_inhib_idx = [self.modules[i].type_of_network == "inhib" for i in range(len(self.modules))]


            modules_ex = [self.modules[i] for i, value in enumerate(modules_ex_idx) if value]
            module_inhib = [self.modules[i] for i, value in enumerate(modules_inhib_idx) if value]
            module_inhib = module_inhib[0]

            for module in modules_ex:
                num_connections_from = module_inhib._N
                self.set_connection(module_inhib, module, weight_scheme, weight_range, scaling_factor, num_connections_from, delay)


if __name__ == "__main__":
    Module_ex = Modules(100, 20, "exc", connections_with_in=1000)
    Modules_inhib = Modules(200, 1, "inhib", connections_with_in=40000)
    Module_ex.set_Connections_within("constant", 1, 17)
    Modules_inhib.set_Connections_within("random", (-1, 0), 1)

    community = Community()
    for i in range(8):
        Module_ex = Modules(100, 20, "exc", connections_with_in=1000)
        Module_ex.set_Connections_within("constant", 1, 17)
        community.modules.append(Module_ex)
    community.modules.append(Modules_inhib)
    community.modules[0].type_of_network

    community.set_connection_btw_modules("Focal", "random", (0, 1), 50, 1)
    community.set_connection_btw_modules("Diffuse", "random", (-1, 0), 2, 1)

    breakpoint()



