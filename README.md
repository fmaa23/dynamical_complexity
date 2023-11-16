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
- `Module` : The class used for constructing each modules within the modular network by inheriting the `iznetwork` class. 

Methods:
    - _set_network_pars_: setting the neurons parameters (a,b,c,d) within a single module
    - _setCurrentWithBackgroundFiring_: setting the background firing for random neurons within a single module 
    - _set_Connections_within_: setting connection between neurons within a single module

- `Connection`: The class represents a connection between modules.

Attributes:
    - module1
    - module2
    - weight
    - delay 

- `Community`

Attributes:

Methods:
    - _set_connection_: setting `Connection` class between modules by 
    - _set_connection_btw_modules_: setting  
    - _make_modular_small_world_:
    - _try_rewiring_:
    - _rewire_: 
    - _generate_final_network_: combining all the modules 
    - _plot_Weights1_: plotting the connectivity matrix and the weight of the connections

### Method
- `Simulating()`: 
