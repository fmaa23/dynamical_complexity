# Coursework: Dynamical Complexity
Done by: Carlos Villalobos Sanchez, Fatima M S Al-ani, Gita A Salsabila

## Files
- [README.md](./README.md): instructions for running the code
- [iznetwork.py](./iznetwork.py): given iznetwork class to simulate 
- [main.py](./main.py): the main program simulating how modular network works  

## Running the code
1. To run the code, use the following command within current directory: `python3 main.py`

## Program Structure 
### Classes
#### `Module` :
The class used for constructing each modules within the modular network by inheriting the `iznetwork` class. 

**Methods**:
- _set_network_pars_: setting the neurons parameters (a,b,c,d) within a single module
- _setCurrentWithBackgroundFiring_: setting the background firing for random neurons within a single module
- _set_Connections_within_: setting connection between neurons within a single module

#### `Connection`
The class represents a connection between modules.


#### `Community`
**Methods**:
- _set_connection_: setting `Connection` class between modules by setting the weight and the delay of the connection
- _set_connection_btw_modules_: setting the  
- _make_modular_small_world_: connectimg 
- _try_rewiring_: 
- _rewire_: rewiring the existing connection by changing the 
- _generate_final_network_: combining all connected the modules to make one big module that become the final network
- _plot_Weights1_: plotting the connectivity matrix and the weight of the connections

### Method
- `Simulating()`: simulating how the final network works during certain duration 'T' by giving 'p' (rewiring probability). This method will shows the **connectivity matrix** to illustrate the structure of the final networks, the **raster plot** to show which parts of the network fires in certain time, and the **mean firing rate** to show number of spikes of each module within each 50ms windows shifted every 20ms.
