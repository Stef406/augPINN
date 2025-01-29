from scipy.constants import pi, g
import numpy as np
import torch

def bed(T, r, z, xo2=0.0):
    # Constants and variables
    P = 101325                                                                  # Operating pressure [Pa] 
    PM_o2 = 32 * 10**-3                                                         # Molecular weight O2, [kg/kmol]
    PM_n2 = 28 * 10**-3                                                         # Molecular weight N2, [kg/kmol]
    PMf = xo2 * PM_o2 + (1 - xo2) * PM_n2                                       # Averaged molecular weight of fluid, [kg/kmol]. Default value is in pyrolysis conditions x=0
    R = 8.314                                                                   # Universal gas constant [J/mol K]
    fw = 0.25                                                                   # Bubble wake fraction for Geldart B, 250 micron size
    d_bed = 0.14                                                                # Bed diameter [m]
    
    # Convert temperature to Kelvin if necessary
    if 500 <= T <= 650:
        T += 273.15                                                             # Convert to absolute temperature [K]
    elif 500 + 273.15 <= T <= 650 + 273.15:
        T = T
    else:
        raise ValueError("Temperature should be in the range of 500 - 650 degC")
    
    # Identify operating conditions (from experiments)
    if T == 500 + 273.15:
        Qmf = 13.4                                                              # Measured minimum fluidisation flow rate at normal conditions [Nlpm]
        mu_f = 35.08e-6                                                         # Viscosity of N2 [Pa s]
        rhof = P * PMf / (R * T)                                                # Density of nitrogen [kg/m3]
    elif T == 600 + 273.15:
        Qmf = 10.4
        mu_f = 38.05e-6                                                         # Viscosity of N2 [Pa s]
        rhof = P * PMf / (R * T)
    elif T == 650 + 273.15:
        Qmf = 10
        mu_f = 39.46e-6                                                         # Viscosity of N2 [Pa s]
        rhof = P * PMf / (R * T)

    if T == 500 + 273.15:
        eps_mf = 0.4649                                                         # Bed voidage at experimental conditions
        if r == 1:
            rho_bed = 1418                                                      # Bed bulk density at experimental condition [kg/m^3]
            eps = eps_mf
            H_bed = 0.164
        elif r == 1.25:
            rho_bed = 1331
            eps = 0.4977
            H_bed = 0.176
        elif r == 1.5:
            rho_bed = 1335
            eps = 0.4961
            H_bed = 0.175
        elif r == 2:
            rho_bed = 1280
            eps = 0.5170
            H_bed = 0.183
    elif T == 600 + 273.15:
        eps_mf = 0.4755
        if r == 1:
            rho_bed = 1390
            eps = eps_mf
            H_bed = 0.168
        elif r == 1.25:
            rho_bed = 1334
            eps = 0.4966
            H_bed = 0.175
        elif r == 1.5:
            rho_bed = 1319
            eps = 0.5020
            H_bed = 0.177
        elif r == 2:
            rho_bed = 1287
            eps = 0.5143
            H_bed = 0.182
    elif T == 650 + 273.15:
        eps_mf = 0.4630
        if r == 1:
            rho_bed = 1423
            eps = eps_mf
            H_bed = 0.164
        elif r == 1.25:
            rho_bed = 1352
            eps = 0.4898
            H_bed = 0.173
        elif r == 1.5:
            rho_bed = 1316
            eps = 0.5032
            H_bed = 0.178
        elif r == 2:
            rho_bed = 1307
            eps = 0.5068
            H_bed = 0.179
    
    # Calculate bed properties
    umf = (Qmf / (pi / 4 * d_bed**2 * 1000 * 60)) * (T / (25 + 273.15))         # Minimum fluidization velocity at operating temperature [m/s]
    u = r * umf                                                                 # Operating gas velocity [m/s]
    
    # Calculate bubble diameters
    db0 = 2.78 / g * (u - umf)**2                                               # Initial bubble diameter [m]
    dm = (0.65 * ((pi/4) * (d_bed * 100)**2 * (u*100 - umf*100))**0.4) / 100    # Limiting size of bubble [m]
    
    # Single value of z instead of an array
    z = np.clip(z, 0, H_bed)
    db = dm - (dm - db0) * np.exp(-0.3 * z/d_bed)                               # Bubble diameter at location z [m]
    ubr = 0.711 * (g * db)**0.5                                                 # Rise velocity for a single bubble at location z [m/s]
    ub = u - umf + ubr                                                          # Rise velocity for bubbles at location z [m/s]
    
    d = np.zeros(len(z),)
    us = np.zeros(len(z),)
    uz = np.zeros(len(z),)
    
    for i in range(len(z)):
        # Calculate fraction of bed in bubbles (d), velocity of sinking solids (us) and net average vertical velocity of bed solids (uz)
        if ub[i] < umf / eps_mf:                                                # Slow bubbles
            d[i] = (u - umf) / (ub[i] + 2 * umf)                                # [m^3 bubbles/m^3 bed]
            d[i] = np.clip(d[i], 0.0001, 0.99)
            us[i] = (fw * d[i] * ub[i]) / (1 - d[i] - d[i] * fw)
            uz[i] = ub[i] - us[i]  # [m/s]
        elif umf / eps_mf < ub[i] < 5 * umf / eps_mf:                           # Intermediate bubbles with thick clouds
            d[i] = ((u - umf) / (ub[i] + umf) + (u - umf) / ub[i]) / 2
            d[i] = np.clip(d[i], 0.0001, 0.99)
            us[i] = (fw * d[i] * ub[i]) / (1 - d[i] - d[i] * fw)
            uz[i] = ub[i] - us[i]  # [m/s]
        elif ub[i] > 5 * umf / eps_mf:                                          # Fast bubbles
            d[i] = (u - umf) / (ub[i] - umf)
            d[i] = np.clip(d[i], 0.0001, 0.99)
            us[i] = (fw * d[i] * ub[i]) / (1 - d[i] - d[i] * fw)
            uz[i] = ub[i] - us[i]  # [m/s]
        elif 6 <= r <= 20:                                                      # Vigorous bubbling
            d[i] = u / ub[i]
            d[i] = np.clip(d[i], 0.0001, 0.99)
            us[i] = (fw * d[i] * ub[i]) / (1 - d[i] - d[i] * fw)
            uz[i] = ub[i] - us[i]  # [m/s]
            
    
    # Derivatives
    def derivative(func, x, h=1e-6):
        return (func(x + h) - func(x - h)) / (2 * h)
    
    dbb = lambda x: (dm - (dm - db0) * np.exp(-0.3 * x / d_bed))
    ubrr = lambda x: (0.711 * np.sqrt(g * dbb(x)))
    ubb = lambda x: u - umf + ubrr(x)
    
    ubb_prime = lambda x: derivative(ubb, x)
    ubb_prime_z = ubb_prime(z)
    
    for _, i in enumerate(z):
        if ubb(i) < umf / eps_mf:                                               # Slow bubbles
            dd = lambda x: (u - umf) / (ubb(x) + 2 * umf)                       # [m^3 bubbles/m^3 bed]
            uss = lambda x: (fw * dd(x) * ubb(x)) / (1 - dd(x) - dd(x) * fw)
            uzz = lambda x: ubb(x) - uss(x)
            
            uzz_prime = lambda x: derivative(uzz, x)
            dd_prime = lambda x: derivative(dd, x)
            
            dd_prime_z = dd_prime(z)
            uzz_prime_z = uzz_prime(z)
        elif umf / eps_mf < ubb(i) < 5 * umf / eps_mf:                          # Intermediate bubbles with thick clouds
            dd = lambda x: ((u - umf) / (ubb(x) + umf) + (u - umf) / ubb(x)) / 2
            uss = lambda x: (fw * dd(x) * ubb(x)) / (1 - dd(x) - dd(x) * fw)
            uzz = lambda x: ubb(x) - uss(x)
            
            uzz_prime = lambda x: derivative(uzz, x)
            dd_prime = lambda x: derivative(dd, x)
            
            dd_prime_z = dd_prime(z)
            uzz_prime_z = uzz_prime(z)
        elif ubb(i) > 5 * umf / eps_mf:                                         # Fast bubbles
            dd = lambda x: (u - umf) / (ubb(x) - umf)
            uss = lambda x: (fw * dd(x) * ubb(x)) / (1 - dd(x) - dd(x) * fw)
            uzz = lambda x: ubb(x) - uss(x)
            
            uzz_prime = lambda x: derivative(uzz, x)
            dd_prime = lambda x: derivative(dd, x)
            
            dd_prime_z = dd_prime(z)
            uzz_prime_z = uzz_prime(z)
        else:                                                                   # Vigorous bubbling
            dd = lambda x: u / ubb(x)
            uss = lambda x: (fw * dd(x) * ubb(x)) / (1 - dd(x) - dd(x) * fw)
            uzz = lambda x: ubb(x) - uss(x)
            
            uzz_prime = lambda x: derivative(uzz, x)
            dd_prime = lambda x: derivative(dd, x)
            
            dd_prime_z = dd_prime(z)
            uzz_prime_z = uzz_prime(z)
    

    # Convert to torch tensor for the PINN
    ub = torch.from_numpy(ub).view(-1, 1)
    uz = torch.from_numpy(uz).view(-1, 1)
    uzz_prime_z = torch.from_numpy(uzz_prime_z).view(-1, 1)
    
    return rho_bed, eps, H_bed, ub, uz, uzz_prime_z, rhof, ubb_prime_z, mu_f