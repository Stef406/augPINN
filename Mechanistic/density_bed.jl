function get_rho_eps(T, r)

    mu_bed = 0.51                                                                   # Fluidized bed viscosity, [Ns/m^2]

    # Convert temperature to Kelvin if necessary
    if 500 <= T <= 650
        T += 273.15                                                                 # Convert to absolute temperature [K]
    elseif 500 + 273.15 <= T <= 650 + 273.15
        T = T
    else
        error("Temperature should be in the range of 500 - 650 degC")
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

    return rho_bed, eps, mu_bed, H_bed
end