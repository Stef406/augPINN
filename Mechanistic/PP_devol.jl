function devol(t, T, d0, rho0, ml)
    # Kinetics of polypropylene pyrolysis in N2 (valid from 500 to 650 degC)
    Ar = 3.3            # Reference pre-exponential parameter [s^-1]
    E = 27500.0         # Apparent activation energy [J/mol]
    psi = 0.87          # Experimental fitting parameter
    R = 8.314           # Universal gas constant [J/mol K]
    P = 101325.0        # Pressure [Pa]
    PM_vm = 128.0       # Naphtalene assumed as lumped components for PP volatiles [kg/kmol]
    w = 0.99            # Volatiles mass fraction in PP  
    X_dev = 0.99        # Final PP conversion at devolatilization time
    d_hole = 1.5e-3     # Hole in the plastic sample [m]
    dref = 8e-3         # Reference initial particle diameter used for the fitting [m]

    # Temperature adjustment
    if 500 <= T <= 650
        T += 273.15     # Convert to absolute temperature [K]
    elseif 500 + 273.15 <= T <= 650 + 273.15
        # T is already in Kelvin, no change needed
    else
        throw(ArgumentError("Temperature should be in the range of 500 - 650 degC"))
    end

    # Kinetics rate constant [s^-1]
    k = Ar * (dref / d0)^psi * exp(-E / (R * T))

    # Compute variables for each element in the vector t
    t_dev = exp(log(-log(1 - X_dev)) - psi * log(Ar * (dref / d0)) + E/(R*T))           # Devolatilization time

    if t < t_dev
        d = (d_hole^3 .+ (d0^3 .- d_hole^3) .* exp.(-k .* t)) .^ (1/3)                  # Taking the hole into account
    else
        d = d_hole
    end
  
    V_eff0 = π/6 * (d0^3 - d_hole^3)
    m = rho0 .* V_eff0 .* exp.(-k .* t)
    rho = (m .+ ml) ./ (π/6 * d.^3)
    rhovm = P * PM_vm / (R * T)
    Q = rho0 * V_eff0 * w / rhovm * k.*exp.(-k .* t)

    return d, rho, rhovm, Q, k
end