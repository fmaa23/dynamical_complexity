# Coursework: Dynamical Complexity
Done by: Carlos Villalobos Sanchez, Fatima M S Al-ani, Gita A Salsabila

## Files
- [README.md](./README.md): instructions for running the code
- [iznetwork.py](./iznetwork.py): given iznetwork class to simulate 
- [main.py](./main.py): the main program simulating how modular network works  

## Running the code
1. To run the code, use the following command within current directory: `python3 main.py`

## Classes & 
- Modules (class)
The class used for constructing each modules within the modular network
    - set_network_pars: method for setting the neurons parameters (a,b,c,d) within a single module
    - setCurrentWithBackgroundFiring: method for setting the background firing for random neurons within a single module 
    - set_Connections_within: method for setting connection between neurons within a single module

- Connection (class)

- Community (class)
    methods:
    - set_connection
    - set_connection_btw_modules
    - make_modular_small_world
    - try_rewiring
    - rewire
    - generate_final_network
    - plot_Weights1

- Simulating (method)
