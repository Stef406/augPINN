# augPINN

Copyright 2025, All Rights Reserved

Code by Stefano Iannello
For Paper, "A hybrid physics-machine learning approach 
            for modelling plastics-bed interactions during fluidized bed pyrolysis"
Energy & Fuels, 2025,
https://pubs.acs.org/doi/10.1021/acs.energyfuels.4c05870,
by S. Iannello, A. Friso, F. Galvanin, M. Materazzi.

This project provides two different PINN frameworks in Python and a purely mechanistic model in Julia to describe the motion of polypropylene (PP) particles within a fluidized bed reactor during pyrolysis.

The main codes are:
  - PINN.py to train and test the PINN.
  - augPINN.py to train and test an augmented version of the PINN.
  - NN.py to train and test a data-driven neural network.
  - force_ODE.jl to solve the force balance to describe the motion of the PP particle.

Useful functions are:
  - density_bed.py and density_bed.jl for some of the physical properties of the fluidized bed.
  - FluidBed.py and bed.jl to model the fluidized bed (based on the K-L model).
  - PP_devol.py and PP_devol.jl to model the devolatilization of the PP particle during pyrolysis (for more information about this topic, check https://doi.org/10.1016/j.cej.2021.133807).
