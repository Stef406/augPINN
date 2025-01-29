using ForwardDiff

function bed(T, r, z, x=0.0)
    # Constants and variables
    P = 101325;                                                                     # [Pa] operating pressure
    PMf = 32 * x + 28 * (1 - x);
    g = 9.81;                                                                       
    fw = 0.3;                                                                       # Bubble wake fraction for Geldart B, 250 micron size
    R = 8.314;                                                                      # Universal gas constant [J/mol K]
    PM_o2 = 32 * 10^(-3);                                                           # Molecular weight O2, [kg/kmol]
    PM_n2 = 28 * 10^(-3);                                                           # Molecular weight N2, [kg/kmol]
    PMf = x * PM_o2 + (1 - x) * PM_n2;                                              # Averaged molecular weight of fluid, [kg/kmol]. Default value is in pyrolysis conditions x=0
    rhof = PMf / (R * T);                                                           # Fluidizing gas density, [kg/m^3]
    muf = (exp(-3.61161) * T^0.740677) * 10^(-5);                                   # Fluidizing gas viscosity, [Ns/m^2]
    mu_bed = 0.51;                                                                  # Fluidized bed viscosity, [Ns/m^2]

    # Convert temperature to Kelvin if necessary
    if 500 <= T <= 650
        T += 273.15;                                                                # Convert to absolute temperature [K]
    elseif 500 + 273.15 <= T <= 650 + 273.15
        T = T;
    else
        error("Temperature should be in the range of 500 - 650 degC")
    end

    # Identify operrating conditions (from experiments)
    if T == 500 + 273.15
        Qmf = 13.4;                                                                 # Measured minimum fluidisation flow rat at normal conditions [Nlpm]
        mu_f = 35.08e-6;                                                            # Viscosity of N2 [Pa s]
        rhof = P * PMf / (R * T);                                                   # Density of nitrogen (kg/m3)
    elseif T == 600 + 273.15
        Qmf = 10.4;
        mu_f = 38.05e-6;                                                            # Viscosity of N2 [Pa s]
        rhof = P * PMf / (R * T);
    elseif T == 650 + 273.15
        Qmf = 10;
        mu_f = 39.46e-6;                                                            # Viscosity of N2 [Pa s]
        rhof = P * PMf / (R * T);
    end 

    if T == 500 + 273.15
        eps_mf = 0.4649;                                                            # Bed vodage at experimental conditions
        if r == 1
            rho_bed = 1418;                                                         # Bed bulk density at experimental condition, [kg/m^3]
            eps = eps_mf;
            H_bed = 0.164;
        elseif r == 1.25
            rho_bed = 1331;
            eps = 0.4977;
            H_bed = 0.176;
        elseif r == 1.5
            rho_bed = 1335;
            eps = 0.4961;
            H_bed = 0.175;
        elseif r == 2
            rho_bed = 1280;
            eps = 0.5170;
            H_bed = 0.183;
        end
    elseif T == 600 + 273.15
        eps_mf = 0.4755;
        if r == 1
            rho_bed = 1390;
            eps = eps_mf;
            H_bed = 0.168;
        elseif r == 1.25
            rho_bed = 1334;
            eps = 0.4966;
            H_bed = 0.175;
        elseif r == 1.5
            rho_bed = 1319;
            eps = 0.5020;
            H_bed = 0.177;
        elseif r == 2
            rho_bed = 1287;
            eps = 0.5143;
            H_bed = 0.182;
        end
    elseif T == 650 + 273.15;
        eps_mf = 0.4630;
        if r == 1
            rho_bed = 1423;
            eps = eps_mf;
            H_bed = 0.164;
        elseif r == 1.25
            rho_bed = 1352;
            eps = 0.4898;
            H_bed = 0.173;
        elseif r == 1.5
            rho_bed = 1316;
            eps = 0.5032;
            H_bed = 0.178;
        elseif r == 2
            rho_bed = 1307;
            eps = 0.5068;
            H_bed = 0.179;
        end
    end

    # Calculate bed properties
    umf = (Qmf / (π / 4 * 0.14^2 * 1000 * 60)) * (T / 273.15);                      # Minimum fluidization velocity at operating temperature, [m/s]
    u = r * umf;                                                                    # Operating gas velocity, [m/s]

    # Calculate bubble diameters
    db0 = (2.78 / g * (u - umf)^2);                                                 # Initial bubble diameter [m]
    dm = (0.65 * (pi / 4 * 14^2 * (u / 100 - umf / 100))^0.4) / 100;                # Limiting size of bubble, [m]

    # Single value of z instead of an array
    z = clamp(z, 0, H_bed)                                                          # Ensure z is not negative
    db = (dm - (dm - db0) * exp(-0.3 * z/0.14));                                    # Bubble diameter at location z, [m]
    ubr = (0.711 * sqrt(g*db));                                                     # Rise velocity for a single bubble at location z, [m/s]
    ub = u - umf + ubr;                                                             # Rise velocity for bubbles at location z, [m/s]
    
    # Derivatives
    dbb(x) = (dm - (dm - db0) * exp(-0.3 * x/0.14));
    ubrr(x) = (0.711 * sqrt(g*dbb(x)));
    ubb(x) = u - umf + ubrr(x);
    ubb_prime(x) = ForwardDiff.derivative(ubb, x);
    dub_dz = ubb_prime(z);


    # Calculate fraction of bed in bubbles (d), velocity of sinking solids (us) and net average vertical velocity of bed solids (uz)
    if ub < umf / eps_mf                                                           # Slow bubbles
        d = (u - umf) / (ub + 2 * umf);     # [m^3 bubbles/m^3 bed]
        d = clamp(d, 0.0001, 0.99);

        dd_slow(x) = (u - umf) / (ubb(x) + 2 * umf);
        uss_slow(x) = (fw * dd_slow(x) * ubb(x)) / (1 - dd_slow(x) - dd_slow(x) * fw);
        uzz_slow(x) = ubb(x) - uss_slow(x);

        uzz_prime_slow(x) = ForwardDiff.derivative(uzz_slow, x);

        duz_dz = uzz_prime_slow(z);

    elseif umf / eps_mf < ub < 5 * umf / eps_mf                                     # Intermediate bubbles with thick clouds
        d = ((u - umf) / (ub + umf) + (u - umf) / ub) / 2;
        d = clamp(d, 0.0001, 0.99);

        dd_inter(x) = ((u - umf) / (ubb(x) + umf) + (u - umf) / ubb(x)) / 2;
        uss_inter(x) = (fw * dd_inter(x) * ubb(x)) / (1 - dd_inter(x) - dd_inter(x) * fw);
        uzz_inter(x) = ubb(x) - uss_inter(x);

        uzz_prime_inter(x) = ForwardDiff.derivative(uzz_inter, x);

        duz_dz = uzz_prime_inter(z);

    elseif ub > 5 * umf / eps_mf                                                    # Fast bubbles
        d = (u - umf) / (ub - umf);
        d = clamp(d, 0.0001, 0.99);

        dd_fast(x) = (u - umf) / (ubb(x) - umf);
        uss_fast(x) = (fw * dd_fast(x) * ubb(x)) / (1 - dd_fast(x) - dd_fast(x) * fw);
        uzz_fast(x) = ubb(x) - uss_fast(x);

        uzz_prime_fast(x) = ForwardDiff.derivative(uzz_fast, x);

        duz_dz = uzz_prime_fast(z);

    else                                                                            # Vigorous bubbling
        d = u / ub;
        d = clamp(d, 0.0001, 0.99);

        dd_vig(x) = u / ubb(x);
        uss_vig(x) = (fw * dd_vig(x) * ubb(x)) / (1 - dd_vig(x) - dd_vig(x) * fw);
        uzz_vig(x) = ubb(x) - uss_vig(x);

        uzz_prime_vig(x) = ForwardDiff.derivative(uzz_vig, x);

        duz_dz = uzz_prime_vig(z);

    end

    us = (fw * d * ub) / (1 - d - d * fw);
    uz = ub - us; # [m/s]

    return uz, ub, rho_bed, H_bed, mu_bed, mu_f, rhof, dub_dz, duz_dz
end