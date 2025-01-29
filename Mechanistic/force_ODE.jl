##Load the differential equations package
using DifferentialEquations, Plots, CSV, DataFrames

## Import useful functions
include("PP_devol.jl");
include("bed.jl");
include("density_bed.jl");

## Define Constants
T = 500;
ml = 0.17e-3;
rhop0 = 900;
dp0 = 12e-3;
g = 9.81;
r = 1.25;
_, _, _, H_bed = get_rho_eps(T, r);


## Define force balance
function force!(du, u, p, t)

    dp, rhop, _, Q, _ = devol(t, T, dp0, rhop0, ml);
    uz, ub, rho_bed, H_bed, mu_bed, mu_f, rho_f, _, duz_dz = bed(T, r, u[2]);

    # Emulsion phase
    ag = -g;
    ab_e = rho_bed / rhop * g;
    Re_e = dp * rhop * abs(uz - u[1])/mu_bed;
    Cd_e = 24/Re_e * (1 + 0.15 * Re_e^0.681) + 0.407/(1 + (8710 / Re_e));
    ad_e = 3/4 * Cd_e * rho_bed * (uz - u[1]) * abs(uz - u[1]) / (rhop * dp);
    av = 1/2 * rho_bed/rhop * (du[1] - uz * duz_dz);
    al = 2.232 * rho_bed * g^0.6 * Q^0.8 / (pi * rhop * dp^2);
    
    # Bubble phase
    ab_b = rho_f / rhop * g;
    Re_b = dp * rhop * abs(ub - u[1])/mu_f;
    Cd_b = 24/Re_b * (1 + 0.15 * Re_b^0.681) + 0.407/(1 + (8710 / Re_b));
    ad_b = 3/4 * Cd_b * rho_f * (ub - u[1]) * abs(ub - u[1]) / (rhop * dp);
    
    if u[2] > H_bed
        du[1] = ag + ab_b + ad_b;
        du[2] = u[1];
    elseif u[2] < 0.0
        du[1] = 0.0;
        du[2] = 0.0;
    else
        du[1] = ag + ab_e + ad_e + av + al;
        du[2] = u[1];
    end

end

u0 = [0.0; H_bed];
tspan = (0.0, 100.0);

prob = ODEProblem(force!, u0, tspan);
sol = solve(prob, TRBDF2(), saveat = 0.1);

# Plot and save the solution
p1 = plot(sol, linewidth = 2, title = "Velocity", 
    xaxis = "t [s]", yaxis = "vp [m/s]", label = false, idxs = (0, 1));

p2 = plot(sol, linewidth = 2, title = "Displacement", 
    xaxis = "t [s]", yaxis = "zp [m]", label = false, idxs = (0, 2));

plot(p1, p2, layout=(2, 1))

CSV.write("results.csv", DataFrame([reshape(sol.t, length(sol.t),1) reshape(sol[2,:]/H_bed, length(sol[2,:]),1)], :auto), header=false)