import random
import numpy as np
from iznetwork import*

import matplotlib.pyplot as plt


# Generating Modules
class Modules(IzNetwork):
    """Subclass of the Iznetwork class
       type_of_network (str): The type of network specified as "inhib" or "exc"
       connections_with_in (int): The number of connections within the module
    """
    def __init__(self, N, Dmax, type_of_network ="exc", connections_with_in=0):
        super().__init__(N, Dmax)
        self.type_of_network = type_of_network
        self.connections_with_in = connections_with_in
        self._N
        self.set_network_pars()

    def set_network_pars(self):

        a_n, b_n, c_n, d_n =[], [], [], []

        for i in range(self._N):
            r = random.uniform(0, 1)
            if self.type_of_network == "exc":
                a, b, c, d = 0.02, 0.2, (-65 + 15 * r ** 2), (8 - 6 * r ** 2)
            elif self.type_of_network == "inhib":
                a, b, c, d = (0.02 + 0.08 * r), (0.25 - 0.05 * r), -50, 2
            else:
                raise ValueError('Network type invalid. Should be "inhib" or "exc"')
            a_n.append(a)
            b_n.append(b)
            c_n.append(c)
            d_n.append(d)

        self.a, self.b, self.c, self.d = a_n, b_n, c_n, d_n
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
        """" Setting connections within a single module. The weights and delays are updated within the module parameters
             using update methods of the IzNetwork class. The weights are given by self._W and the delays as self._D

             Args:
                 weight_scheme (str): The weight scheme through which the neurons within the module is connected;
                                      "constant" or "random"
                 weight_range (tuple): The range through which the weights are to be taken randomly,
                                       if constant single number should be given
                 scaling_factor (int): The scaling factor of the connections between neurons

              """

        # Connections Array
        connection_size = (self._N,self._N)

        # Initializing a network connection matrix with disconnected neurons
        connected_neurons = np.full(connection_size, False)

        # Random connections
        num_connections_with_in = self.connections_with_in
        indices = set()

        while len(indices) < num_connections_with_in:
            new_index = (random.randint(0, connection_size[0]-1), random.randint(0, connection_size[1]-1))
            if new_index[0] != new_index[1]:
                indices.add(new_index)

        indices = list(indices)

        for index in indices:
            connected_neurons[index[0]][index[1]] = True

        # Updating weights based on the specified weighting scheme
        if weight_scheme == "constant":
            weight = weight_range[0]
        elif weight_scheme == "random":
            weight = np.random.uniform(weight_range[0], weight_range[1], size=num_connections_with_in)
        else:
            raise ValueError('Scheme invalid. Should be "constant" or "random"')

        scaled_weight = weight * scaling_factor

        # Initializing disconnected network weights
        weights = np.zeros((self._N,self._N))

        # Updating connected neurons weights
        weights[connected_neurons] = weights[connected_neurons] + scaled_weight
        self.setWeights(weights)

        # Initializing disconnected network delays
        delays = np.zeros((self._N,self._N))
        delays = delays.astype(int)

        random_integers = np.random.randint(1, self._Dmax+1, size=num_connections_with_in)

        # Updating connected neurons delays
        delays[connected_neurons] += random_integers
        self.setDelays(delays)

class Connection():
    """The connection class represents a single between-modules connection. The connection attributes include
       module1(Module): The module from which the connections start
       module2(Module): The module to which the connections are directed
       weights(np.array): The weights of the connection
       delays(np.array): The delays of the connection
    """
    def __init__(self, module1=None, module2=None, weights=None, delays=None):
        self.module1 = module1
        self.module2 = module2
        self.weights = weights
        self.delays = delays

class Community():
    """The community class represents a community with modules and the connections between modules.
       modules(list): List containing all models in the community
       connections(list): List of all connections between modules

       Kindly note that the connections attribute only include between modules connections not within modules.
       Connections within modules are specified in the module (Module class)attributes via the weights array module._W
       and delays array module._D
    """
    def __init__(self, modules=None, connections=None):
        # List of modules present in the community
        if modules is None:
            self.modules = []  # Create a new list for each instance
        else:
            self.modules = modules
        if connections is None:
            self.connections = []  # Create a new list for each instance
        else:
            self.connections = connections
        # final_network = None

    def set_connection(self,module1, module2, weight_scheme, weight_range, scaling_factor, num_connections_from, delay,
                       connections_to_all = True, start_connections_idx = None, end_connections_idx= None):
        """Sets a connection between two modules.
           Args:
               module1 (Module):     The module the connections starts from
               module2 (Module):     The module the connections ends in
               weight_scheme (str):  The weight scheme through which the neurons within the module is connected;
                                      "constant" or "random"
               weight_range (tuple): The range through which the weights are to be taken randomly,
                                       if constant single number should be given
               scaling_factor (int): The scaling factor of the connections between neurons
               num_connections_from (int): The number of neurons participating in the connection from module 1
               delay (int): The maximum delay of the connection in milliseconds
               connections_to_all (bool): Connections to be to all neurons in the second module
               start_connections_idx (int): If connections_random is False, specify start connection index
               end_connections_idx (int): If connections_random is False, specify end connection index

        """

        # Connections Array
        connection_size = (module1._N,module2._N)
        connected_neurons = np.full(connection_size, False)

        # Random connections
        indices = np.random.choice(connection_size[0], num_connections_from, replace=False)

        # Connecting from selected neuron indices to all neurons in the second module
        if connections_to_all:
            connected_neurons[indices,:] = True
        else:
            connected_neurons[indices, start_connections_idx:end_connections_idx] = True

        # Getting weights based on the specified weighting scheme
        if weight_scheme == "constant":
            weight = weight_range[0]
        elif weight_scheme == "random":
            # Size is number of neurons the connection is from to number of neurons the connection is directed to
            if connections_to_all:
                weight = np.random.uniform(weight_range[0], weight_range[1], size=num_connections_from * module2._N)
            else:
                weight = np.random.uniform(weight_range[0], weight_range[1], size=num_connections_from * (end_connections_idx-start_connections_idx))
        else:
            raise ValueError('Scheme invalid. Should be "constant" or "random"')

        # Initializing weights for disconnected network
        weights = np.zeros((module1._N,module2._N))

        # Updating weights of connected neurons
        scaled_weight = weight * scaling_factor
        weights[connected_neurons] = weights[connected_neurons] + scaled_weight

        # Specifying a connection with its modules and weights
        connection = Connection()
        connection.module1 = module1
        connection.module2 = module2
        connection.weights = weights

        # Initializing weights for disconnected network
        delays = np.zeros((module1._N,module2._N))
        delays = delays.astype(int)

        # Updating delays of connected neurons
        if connections_to_all:
            random_integers = np.random.randint(1, delay+1, size=num_connections_from*module2._N)
        else:
            random_integers = np.random.randint(1, delay + 1, size= num_connections_from * (end_connections_idx-start_connections_idx))
        delays[connected_neurons] += random_integers

        # Specifying the connection delays
        connection.delays = delays

        # Appending the connection to the community connections between modules
        self.connections.append(connection)

    def set_connection_btw_modules(self, projection_pattern, weight_scheme, weight_range, scaling_factor, delay):
        """Sets the connection between modules as specified by the project specs.
           projection_pattern (str): "Focal" from specific number of neurons in module 1 to all neurons in module 2
                                     "Diffuse" all to all connection from module 1 to module 2
           weight_scheme (str): The weight scheme through which the neurons within the module is connected;
                                "constant" or "random"
           weight_range (tuple): The range through which the weights are to be taken randomly,
                                 if constant single number should be given
           scaling_factor (int): The scaling factor of the connections between neurons
           delay (int): The maximum delay of the connection in milliseconds

        """

        # Getting excitatory modules in the community
        modules_ex_idx = [self.modules[i].type_of_network == "exc" for i in range(len(self.modules))]
        modules_ex = [self.modules[i] for i, value in enumerate(modules_ex_idx) if value]

        # Getting inhibitory modules in the community
        modules_inhib_idx = [self.modules[i].type_of_network == "inhib" for i in range(len(self.modules))]
        module_inhib = [self.modules[i] for i, value in enumerate(modules_inhib_idx) if value]
        module_inhib = module_inhib[0]

        # Projection schemes as specified by the project specs
        # Focal connection between one random excitatory module and the inhibitory module
        if projection_pattern == "Focal":

            for i, module_ex in enumerate(modules_ex):
                num_connections_from = 4

                start_edges_ind = i*25
                end_edges_ind = (i+1) * 25
                self.set_connection(module_ex, module_inhib, weight_scheme, weight_range, scaling_factor,
                                    num_connections_from, delay, connections_to_all=False,
                                    start_connections_idx=start_edges_ind, end_connections_idx=end_edges_ind)

            # Setting the connection

        # Diffuse connection between the inhibitory module and all neurons in all excitatory modules
        elif projection_pattern == "Diffuse":

            # Looping through all excitatory modules
            for module in modules_ex:
                # All neurons in the inhibitory model are in connection
                num_connections_from = module_inhib._N
                # Setting the connection
                self.set_connection(module_inhib, module, weight_scheme, weight_range, scaling_factor, num_connections_from, delay)
        else:
            raise ValueError('Projection scheme invalid. Should be "Focal" or "Diffuse"')  

    def make_modular_small_world(self, p=0.4):
        import random
        exc_modules = []

        self.rewired_W = np.zeros((800, 800))
        self.rewired_D = np.zeros((800, 800))
        for module in self.modules:
            if module.type_of_network == "exc":
                exc_modules.append(module)
        for origin_module, module in enumerate(exc_modules):
            for i in range(0, 100):
                for j in range(0, 100):
                    if module._W[i][j] > 0:
                        self.try_rewiring(p, random, module, origin_module, i, j)

    def try_rewiring(self, p, random, module, origin_module, i, j):
        """ Contains the logic behind rewiring"""
        rewire_p = random.random()
        if rewire_p <= p:
            # Rewire accordingly
            self.rewire(random, module, origin_module, i, j)
        else:
            # Add the connection to the rewired representation 
            self.rewired_W[origin_module*100 + i][origin_module*100 + j] = module._W[i][j]
            self.rewired_D[origin_module*100 + i][origin_module*100 + j] = module._D[i][j]

    def rewire(self, random, module, origin_module, i, j):
        "Method used in make_modular_small_world"
        rewired = False
        while not rewired:
            random_module = random.randint(0, 6)
            if random_module >= origin_module:
                random_module += 1
            random_neuron = random.randint(0, 99)
            # Rewire the neuron
            if self.rewired_W[origin_module*100 + i][random_module*100 + random_neuron] == 0:
                self.rewired_W[origin_module*100 + i][random_module*100 + random_neuron] = module._W[i][j]
                self.rewired_D[origin_module*100 + i][random_module*100 + random_neuron] = module._D[i][j]
                module._W[i][j] = 0
                module._D[i][j] = 0
                rewired = True

    def generate_final_network(self):
        self.final_network = IzNetwork(1000, 20)

        # set weights adn delays in the final network
        #Initialize weights and delays:
        final_weights = np.zeros((1000, 1000))
        final_delays = np.zeros((1000, 1000), dtype=int)
        # add rewired to final:
        final_weights[0:800, 0:800] = self.rewired_W
        final_delays[0:800, 0:800] = self.rewired_D

        # add excitatory to inhibitory and inhibitory to exhitatory
        i = 0
        j = 0
        for connection in self.connections:
            if(connection.module1.type_of_network == "exc"):
                final_weights[i*100:(i+1)*100, 800:1000] = connection.weights
                final_delays[i*100:(i+1)*100, 800:1000] = connection.delays
                i += 1
            else:
                final_weights[800:1000, j*100:(j+1)*100] = connection.weights
                final_delays[800:1000, j*100:(j+1)*100] = connection.delays
                j += 1
        
        # add inhibitory to inhibitory
        for module in self.modules:
            if module.type_of_network == "inhib":
                #print(np.copy(module._W).shape)
                #print(final_weights[800:1000, 800:1000].shape)
                final_weights[800:1000, 800:1000] = np.copy(module._W)
                print(final_weights[800:1000, 800:1000])
                final_delays[800:1000, 800:1000] = module._D
        self.final_network.setWeights(final_weights)
        self.final_network.setDelays(final_delays)

        self.final_weights = final_weights
        #Initialize parameters:
        final_a = np.zeros(1000)
        final_b = np.zeros(1000)
        final_c = np.zeros(1000)
        final_d = np.zeros(1000)
        # add parameters
        i = 0
        for module in self.modules:
            if(module.type_of_network == "exc"):
                final_a[i*100:(i+1)*100] = module.a
                final_b[i*100:(i+1)*100] = module.b
                final_c[i*100:(i+1)*100] = module.c
                final_d[i*100:(i+1)*100] = module.d
                i += 1
            else:
                final_a[800:1000] = module.a
                final_b[800:1000] = module.b
                final_c[800:1000] = module.c
                final_d[800:1000] = module.d
        self.final_network.setParameters(a=final_a, b=final_b, c=final_c, d=final_d)
    
    def plot_Weights1(self):
        

        # Create a mask: 0 where weights are 0, 1 where weights are non-zero
        mask = np.where(self.final_weights == 0, 0, 1)
        # Create the plot
        plt.imshow(mask, cmap='coolwarm', interpolation='nearest')

        # Optional: Add a color bar
        plt.colorbar()

        # Optional: Add title and labels
        plt.title('Heatmap of Weights')
        plt.xlabel('Neuron Index')
        plt.ylabel('Neuron Index')

        # Show the plot
        plt.show()
    
def simulating(Community, p, T=1000):
    Community.make_modular_small_world(p)
    Community.generate_final_network()
    Community.plot_Weights1()
    
    V = np.zeros((T, community.final_network._N))
    for t in range(T):
        community.final_network.setCurrentWithBackgroundFiring()
        community.final_network.update()
        V[t,:], _ = community.final_network.getState()

    neurons_to_plot = 800
    neurons = np.arange(neurons_to_plot)
    firing_times = V[:, :neurons_to_plot]

    firing_instances, fired_neurons = np.where(firing_times > 29)

    plt.figure(figsize=(10,8))
    plt.subplots_adjust(hspace=0.3, wspace=0.4)

    plt.subplot(2,1,1)
    plt.scatter(firing_instances, fired_neurons)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title('Firing Neurons when p={}'.format(p))

    window_size=50
    shift = 20
    spikes = np.zeros((50,800))
    num_windows = T//window_size

    for j in range(800):
        tuple_array = np.array(list(zip(firing_instances, fired_neurons)))
        time_indices = np.where(fired_neurons==j)
        timings=[]
        padding = window_size - shift

            # Adjust the range to include padding at the start
        for i, t in enumerate(range(-padding, T - window_size + 1, shift)):
            start_time = max(0, t)  # Ensure the start time doesn't go below 0
            end_time = min(t + window_size, T)  # Ensure the end time doesn't exceed T
            timings.append((start_time, end_time))

        times_of_this_neuron_firing = firing_instances[np.where(fired_neurons == j)]
        for k,times in enumerate(timings):
            for time_of_this_neuron_firing in times_of_this_neuron_firing:
                if time_of_this_neuron_firing in range(times[0],times[1]+1):
                    spikes[k,j] = spikes[k,j]+1

    firing=[]
    for i in range(8):
        firing.append(np.sum(spikes[:, i*100: (i+1)*100], axis=1)/100)

    time_points = np.linspace(0, 1000, len(firing[0]))
    for i, arr in enumerate(firing):
        plt.subplot(2,1,2)
        plt.plot(time_points, arr)

    plt.xlabel('Time (ms)')
    plt.ylabel('Mean Firing Rate')
    plt.title('Firing Rate over 1000ms (50 Windows)')

    plt.show()

if __name__ == "__main__":
    # Excitatory neurons network
    Module_ex = Modules(100, 20, "exc", connections_with_in=1000)
    # Inhibitory neurons network
    Modules_inhib = Modules(200, 1, "inhib", connections_with_in=39800)

    # Connections within the excitatory network
    Module_ex.set_Connections_within("constant", (1,), 17)
    # Connections within the inhibitory network
    Modules_inhib.set_Connections_within("random", (-1, 0), 1)

    # Initializing a community
    community = Community()

    # Adding eight excitatory networks to the community
    for i in range(8):
        Module_ex = Modules(100, 20, "exc", connections_with_in=1000)
        Module_ex.set_Connections_within("constant", (1,), 17)
        community.modules.append(Module_ex)

    # Adding one inhibitory networks to the community
    community.modules.append(Modules_inhib)

    # Setting excitatory-inhibitory connections
    community.set_connection_btw_modules("Focal", "random", (0, 1), 50, 1)
    # Setting inhibitory-excitatory connections
    community.set_connection_btw_modules("Diffuse", "random", (-1, 0), 2, 1)

    P = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for p in P:
        simulating(community, p,T=1000)
    
    breakpoint()




