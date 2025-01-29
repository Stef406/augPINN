import numpy as np
import torch

def devol(t, T, d0, rho0, ml):
    # Constants
    Ar = 3.3                                                                    # Reference pre-exponential parameter [s^-1]
    E = 27500.0                                                                 # Apparent activation energy [J/mol]
    psi = 0.87                                                                  # Experimental fitting parameter
    R = 8.314                                                                   # Universal gas constant [J/mol K]
    P = 101325.0                                                                # Pressure [Pa]
    PM_vm = 128.0                                                               # Naphtalene assumed as lumped components for PP volatiles [kg/kmol]
    w = 0.99                                                                    # Volatiles mass fraction in PP  
    X_dev = 0.99                                                                # Final PP conversion at devolatilization time
    d_hole = 1.5e-3                                                             # Hole in the plastic sample [m]
    dref = 8e-3                                                                 # Reference initial particle diameter used for the fitting [m]

    # Temperature adjustment
    if 500 <= T <= 650:
        T += 273.15                                                             # Convert to absolute temperature [K]
    elif 500 + 273.15 <= T <= 650 + 273.15:
        pass                                                                    # T is already in Kelvin, no change needed
    else:
        raise ValueError("Temperature should be in the range of 500 - 650 degC")

    # Kinetics rate constant [s^-1]
    k = Ar * (dref / d0)**psi * np.exp(-E / (R * T))

    # Devolatilization time (calculated)
    t_dev = np.exp(np.log(-np.log(1 - X_dev)) - psi * np.log(Ar * (dref / d0)) + E / (R * T))

    for i in range(len(t)):
        if t[i] < t_dev:
            d = (d_hole**3 + (d0**3 - d_hole**3) * np.exp(-k * t))**(1/3)
        else:
            d = d_hole

    V_eff0 = np.pi / 6 * (d0**3 - d_hole**3)
    m = rho0 * V_eff0 * np.exp(-k * t)
    rho = (m + ml) / (np.pi / 6 * d**3)
    rhovm = P * PM_vm / (R * T)
    Q = rho0 * V_eff0 * w / rhovm * k * np.exp(-k * t)
    
    Q = torch.from_numpy(Q).view(-1, 1)
    d = torch.from_numpy(d).view(-1, 1)
    rho = torch.from_numpy(rho).view(-1, 1)
    
    return d, rho, rhovm, Q, k