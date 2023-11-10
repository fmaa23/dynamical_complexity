import unittest

import random
import numpy as np
from iznetwork import*
from Generating_Networks import Modules, Community

class TestGeneratingNetwork(unittest.TestCase):
    def test_rewiring(self):
        # Excitatory neurons network
        Module_ex = Modules(100, 20, "exc", connections_with_in=1000)
        # Inhibitory neurons network
        Modules_inhib = Modules(200, 1, "inhib", connections_with_in=40000)

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
        community.make_modular_small_world(0.2)
        count = np.sum(community.rewired_W > 0)
        assert count == 8000