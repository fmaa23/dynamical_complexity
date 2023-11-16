import unittest

import random
import numpy as np
from iznetwork import*
from Generating_Networks import Modules, Community

class TestGeneratingNetwork(unittest.TestCase):
    def setUp(self):
        # Excitatory neurons network
        Module_ex = Modules(100, 20, "exc", connections_with_in=1000)
        # Inhibitory neurons network
        Modules_inhib = Modules(200, 1, "inhib", connections_with_in=39800)

        # Connections within the excitatory network
        Module_ex.set_Connections_within("constant", (1,), 17)
        # Connections within the inhibitory network
        Modules_inhib.set_Connections_within("random", (-1, 0), 1)

        # Initializing a community
        self.community = Community()

        # Adding eight excitatory networks to the community
        for i in range(8):
            Module_ex = Modules(100, 20, "exc", connections_with_in=1000)
            Module_ex.set_Connections_within("constant", (1,), 17)
            self.community.modules.append(Module_ex)

        # Adding one inhibitory networks to the community
        self.community.modules.append(Modules_inhib)

        # Setting excitatory-inhibitory connections
        self.community.set_connection_btw_modules("Focal", "random", (0, 1), 50, 1)
        # Setting inhibitory-excitatory connections
        self.community.set_connection_btw_modules("Diffuse", "random", (-1, 0), 2, 1)

    def test_make_modular_small_world(self):
        self.community.make_modular_small_world(0.2)
        count = np.sum(self.community.rewired_W > 0)
        assert count == 8000

    def test_generate_final_network(self):
        self.community.make_modular_small_world(0.2)
        count = np.sum(self.community.rewired_W > 0)
        assert count == 8000

        self.community.generate_final_network()

    def test_plot_weights(self):
        self.community.make_modular_small_world(0.2)
        count = np.sum(self.community.rewired_W > 0)
        assert count == 8000

        self.community.generate_final_network()

        self.community.plot_connections()