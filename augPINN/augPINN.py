#%% Packages
import sys
sys.path.append(r'C:\Users\Stefano\OneDrive - University College London\Desktop\GitHub Repos\augPINN')   
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from FluidBed import bed
from PP_devol import devol
from density_bed import get_rho_eps
from scipy.constants import g
import os
import time  
from sklearn.utils import shuffle
from torchmetrics.regression import R2Score
import copy


#%% ANN class
class FCN(nn.Module):

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.ReLU
        self.fcs = nn.Sequential(
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation())
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        
        return x
    

#%% loss functions
# Define the force balance equation residual
def physics_loss(model, t, T, r):
    
    # Parameters
    ml = 0.17e-3
    d0 = 12e-3
    rho0 = 900
    
    
    # physics loss
    u = model(t)
    u1 = u[:, 0]                                                                # Position
    u2 = u[:, 1]                                                                # Distribution parameter P  

    
    rho_bed, _, H_bed, ub, uz, duzdz, rhof, _, muf = bed(T, r, u1.detach().numpy())
    dp, rhop, rhovm, Q, k = devol(t.detach().numpy(), T, d0, rho0, ml)
    _, _, _, mu_bed = get_rho_eps(T, r)
    
    u1 = torch.clamp(u1, 0.0, H_bed)
    u2 = torch.clamp(u2, 0.0, 1.0)
    
    # Position
    dudt = torch.autograd.grad(u1, t, torch.ones_like(u1), create_graph=True)[0]
    d2udt = torch.autograd.grad(dudt, t, torch.ones_like(dudt), create_graph=True)[0]
    

    # Clamp gradients to avoid instabilities
    dudt = torch.clamp(dudt, -1e5, 1e5)
    d2udt = torch.clamp(d2udt, -1e5, 1e5)
    
    # # Emuslion
    ag = -g
    ab = rho_bed / rhop * g
    
    Re = dp * rhop * torch.abs(uz - dudt) / mu_bed
    Cd = 24/Re * (1 + 0.15 * Re**0.681) + 0.407/(1 + (8710 / Re))
            
    ad = 3/4 * Cd * rho_bed * (uz - dudt) * torch.abs(uz - dudt) / (rhop * dp)
    av = 1/2 * rho_bed/rhop * (d2udt - uz * duzdz)
    al = 2.232 * rho_bed * g**0.6 * Q**0.8 / (np.pi * rhop * dp**2)
    
    # Bubble
    ab_b = rhof / rhop * g
    Re_b = dp * rhof * torch.abs(ub - dudt) / muf
    Cd_b = 24/Re_b * (1 + 0.15 * Re_b**0.681) + 0.407/(1 + (8710 / Re_b))
            
    ad_b = 3/4 * Cd_b * rhof * (ub - dudt) * torch.abs(ub - dudt) / (rhop * dp)
    
    residual = d2udt - ag - (ab + ad + al - av)*u2 - (ab_b + ad_b)*(1 - u2)
    loss_physics = torch.mean(residual**2)

    return loss_physics


def data_loss(model, t, u_exact):
    
    u = model(t)
    
    loss_data = torch.mean((u - u_exact)**2)
    
    return loss_data

#%% Plotting functions
def live_plotLog(file_path, x_vals, y_vals, title, ylabel, xlabel='Iteration', y_labels= None):
    plt.figure(figsize=(10, 6))
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    for i in range(len(y_vals)):
        if len(y_vals) > 1:
            plt.semilogy(x_vals, y_vals[i], label=y_labels[i])
            plt.legend()
        else:
            plt.semilogy(x_vals, y_vals[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.show()
    
def live_plot(file_path, x_vals, y_vals, title, ylabel, xlabel='Iteration', y_labels=None):
    plt.figure(figsize=(10, 6))
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    for i in range(len(y_vals)):
        if len(y_vals) > 1:
            plt.plot(x_vals, y_vals[i], label=y_labels[i])
            plt.legend()
        else:
            plt.plot(x_vals, y_vals[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.show()

#%% Import data and pre-processing
n_exp = 20
data = []
times = []
positions = []

for i in range(n_exp):
    path = os.path.join("C:/Users/Stefano/OneDrive - University College London/Desktop/GitHub Repos/augPINN/data/PP_Pyr/All_clean", f"R{i+1}.xlsx")
    
    # Read and clean data once
    df = pd.read_excel(path).dropna(subset=['t (s)', 'y (cm)', 'T', 'FI'])
    
    # Convert data to required format and scale y (cm)
    data_array = np.transpose(np.array([
        df['t (s)'].values, 
        df['y (cm)'].values / 100,                                              # Convert to meters
        df['T'].values,
        df['FI'].values
    ]))

    # Append data and individual columns
    data.append(data_array)
    times.append(data_array[:, 0])                                              # First column for times
    positions.append(data_array[:, 1:])                                         # Remaining columns for positions and other values
    
# TAKE ONLY THE EXPERIMENTS OF INTEREST
positions = positions[6:9]
times = times[6:9]


#%% Take average of the runs (change code according to number of experiments)

max_len = max(len(positions[0]), len(positions[1]), len(positions[2]))
t = max(times, key = len)

# Create new arrays with NaNs filled up to max_len
padded_run1 = np.pad(positions[0][:, 0], (0, max_len - len(positions[0])), constant_values=np.nan)
padded_run2 = np.pad(positions[1][:, 0], (0, max_len - len(positions[1])), constant_values=np.nan)
padded_run3 = np.pad(positions[2][:, 0], (0, max_len - len(positions[2])), constant_values=np.nan)

# Take the element-wise mean, ignoring NaNs
average_pos = np.nanmean([padded_run1, padded_run2, padded_run3], axis=0)
std_dev = np.nanstd([padded_run1, padded_run2, padded_run3], axis=0)


#%% Train/test split (80 - 20)

dataset = shuffle(np.hstack((np.hstack(t).reshape(-1, 1), np.vstack(average_pos))), random_state=92)

train = dataset[:int(len(dataset)*0.8), :]
test = dataset[int(len(dataset)*0.8):, :]


#%% Convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t_obs = torch.tensor(train[:, 0], dtype=torch.float32).view(-1, 1).to(device)
u_obs = torch.tensor(train[:, 1], dtype=torch.float32).view(-1, 1).to(device)
t_obsTest = torch.tensor(test[:, 0], dtype=torch.float32).view(-1, 1).to(device)
u_obsTest = torch.tensor(test[:, 1], dtype=torch.float32).view(-1, 1).to(device)
t_physics = torch.tensor(train[:, 0], dtype=torch.float32).view(-1, 1).requires_grad_(True).to(device)
t_physicsTest = torch.tensor(test[:, 0], dtype=torch.float32).view(-1, 1).requires_grad_(True).to(device)

# For plotting
t_plot = torch.tensor(max(times, key=len), dtype=torch.float32).view(-1, 1).to(device)

# Extract the directory from the file path (change path as needed)
path = r"C:/Users/Stefano/OneDrive - University College London/Desktop/GitHub Repos/augPINN/data/PP_Pyr/"
output_dir = os.path.join(os.path.dirname(path), 'output_images_500-2_augPINN')
os.makedirs(output_dir, exist_ok=True)                                          # Create directory if it doesn't exist


#%% Training
torch.manual_seed(92)

# Initialize pinn
pinn = FCN(1, 2, 64, 4).to(device)

# Initialize the adaptive weights for the loss terms
e_f = torch.nn.Parameter(torch.tensor(10.0, requires_grad=True, dtype=torch.float32).to(device))
e_d = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, dtype=torch.float32).to(device))


# Update the optimizer to include the adaptive weights
optimiser = torch.optim.Adam(list(pinn.parameters()) + [e_f, e_d], lr=1e-3)


# Early stopping parameters (if needed)
# best_val_loss = float('inf')
# early_stopping_counter = 0
# patience = 200                                                                  # Early stopping patience
# tolerance = 1e-4                                                                # Tolerance for early stopping

# Lists to record useful parameters
epoch_list = []
loss_phy_list = []
loss_data_list = []
loss_phy_test_list = []
loss_data_test_list = []
loss_train_list = []
loss_test_list = []
ef_list = []
ed_list = []
wf_list = []
wd_list = []
r2_train_list = [] 
r2_test_list = []
mse_train_list = []
mse_test_list = []
pinn_param = []

iterations = 10000

metric = R2Score()

# Scalar inputs for the physics loss (change according to experimental conditions)
T = 500
r = 2

start_time = time.time()                                                        # Capture the start time

# Training loop
for i in range(iterations):
    optimiser.zero_grad()
    
    loss_phy = physics_loss(pinn, t_physics, T, r)
    
    loss_data = data_loss(pinn, t_obs, u_obs)
    

    # Compute the loss weights
    w_f = 0.5 / e_f.pow(2)
    w_d = 0.5 / e_d.pow(2)
    sf = torch.log(1 + e_f.pow(2))                                              # To avoid negative penalties and losses
    sd = torch.log(1 + e_d.pow(2))
        
    # Combine losses
    loss = w_f * loss_phy + w_d * loss_data + sf + sd
    
    # Backpropagation
    loss.backward()
    optimiser.step()
    
    # Save the current parameters
    pinn_param.append(copy.deepcopy(pinn.state_dict()))
    
    # Testing
    loss_phyTest = physics_loss(pinn, t_physicsTest, T, r)
    
    loss_dataTest = data_loss(pinn, t_obsTest, u_obsTest)
    
    loss_test = w_f * loss_phyTest + w_d * loss_dataTest + sf + sd
    
    
    # Record losses and other parameters
    epoch_list.append(i + 1)
    loss_phy_list.append(loss_phy.item())
    loss_data_list.append(loss_data.item())
    loss_phy_test_list.append(loss_phyTest.item())
    loss_data_test_list.append(loss_dataTest.item())
    loss_train_list.append(loss.item())
    loss_test_list.append(loss_test.item())
    ef_list.append(e_f.item())
    ed_list.append(e_d.item())
    wf_list.append(w_f.item())
    wd_list.append(w_d.item())
    
    # Make predictions
    with torch.no_grad():
        u_pred_train = pinn(t_obs).detach()[:, 0].view(-1, 1)
        u_pred_test = pinn(t_obsTest).detach()[:, 0].view(-1, 1)
    r2_train_list.append(metric(u_pred_train, u_obs).item())
    r2_test_list.append(metric(u_pred_test, u_obsTest).item())
    mse_train_list.append(torch.mean((u_pred_train - u_obs)**2).item())
    mse_test_list.append(torch.mean((u_pred_test - u_obsTest)**2).item())
    
    
    # # Early stopping check (if needed)
    # if loss_test < best_val_loss - tolerance:
    #     best_val_loss = loss_test
    #     early_stopping_counter = 0
    # else:
    #     early_stopping_counter += 1

    # if early_stopping_counter >= patience:
    #     print(f"Early stopping at epoch {i + 1}")
    #     break
    
    
    if i % 100 == 0:

        print(f"Iteration {i}: Loss train = {loss.item()}, Loss test = {loss_test.item()}")
     
        # Make overall predictions for plotting
        _, H_bed, _, _ = get_rho_eps(T, r)
        with torch.no_grad():
            u = pinn(t_plot).detach()
        u_pred = torch.clamp(u[:, 0].view(-1, 1), 0.0, H_bed)
        P = torch.clamp(u[:, 1].view(-1, 1), 0.0, 1.0)
    
        
        # Plot the loss curves and parameters
        file_path = os.path.join(output_dir, "Losses.png")
        live_plotLog(file_path, epoch_list, [loss_phy_list, loss_data_list, loss_phy_test_list, loss_data_test_list],
                  title='Losses', ylabel='Loss', 
                  y_labels=['Physics loss (train)', 
                            'Data loss (train)', 'Physics loss (test)', 'Data loss (test)'])
        
        file_path = os.path.join(output_dir, "LossesTot.png")
        live_plotLog(file_path, epoch_list, [loss_train_list, loss_test_list],
                  title='Total losses', ylabel='Loss', 
                  y_labels=['Training', 'Testing'])
        
        file_path = os.path.join(output_dir, "AdaptW.png")
        live_plotLog(file_path, epoch_list, [wf_list, wd_list], 
                     title='Adaptive Weights', 
                     ylabel='w', y_labels=['$w_f$', '$w_d$'])
        
        file_path = os.path.join(output_dir, "Uncer.png")
        live_plotLog(file_path, epoch_list, [ef_list, ed_list], 
                     title='Uncertainties', 
                     ylabel=r'$\epsilon$', y_labels=[r'$\epsilon_f$', r'$\epsilon_d$'])
        
        
        file_path = os.path.join(output_dir, "MSE.png")
        live_plotLog(file_path, epoch_list, [mse_train_list, mse_test_list], 
                     title='MSE', 
                     ylabel='MSE', y_labels=['Train', 'Test'])
        
        
        # Plot the figure of axial position
        plt.figure(figsize=(6, 5))
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.scatter(t_plot, average_pos/H_bed, color='black', label='Experiments', s=5)  # Scatter plot for the average
        plt.fill_between(t, (average_pos - std_dev)/H_bed, (average_pos + std_dev)/H_bed, color='gray', alpha=0.3)  # Error band
        plt.plot(t_plot, u_pred/H_bed, label="PINN solution", color="red")
        plt.xlabel("t [s]")
        plt.ylabel(r"$Z_p$ [-]")
        plt.ylim(-0.01, 1.01)
        plt.title(f"Iterations = {i}")
        plt.legend()
        # Save the figure in high quality as PNG
        file_path = os.path.join(output_dir, f"displac_{i}.png")
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        
        # Plot the figure of distribution parameter P
        plt.figure(figsize=(6, 5))
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.plot(t_plot, P, color="black")
        plt.xlabel("t [s]")
        plt.ylabel(r"$P$ [-]")
        plt.title(f"Iterations = {i}")
        # Save the figure in high quality as PNG
        file_path = os.path.join(output_dir, f"P_param_{i}.png")
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        
        # Parity plot
        plt.figure(figsize=(10, 10))
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.scatter(u_obs, u_pred_train, color="black", alpha=0.5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Parity plot (Iterations: {i})")
        min_val = min(min(u_obs), min(u_pred_train))
        max_val = max(max(u_obs), max(u_pred_train))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=2)
        plt.text(x=min_val+0.002, y=max_val-0.002, s=f'$R^2$ = {r2_train_list[i]:.3f}', fontsize=18, 
                 verticalalignment='top', horizontalalignment='left')
        plt.xlim([min_val, max_val])
        plt.ylim([min_val, max_val])
        file_path = os.path.join(output_dir, f"Parity_train{i}.png")
        plt.savefig(file_path, dpi=600, bbox_inches='tight')
        

end_time = time.time()                                                          # Capture the end time
elapsed_time = end_time - start_time                                            # Calculate the elapsed time

# Convert elapsed time to format (hours, minutes, seconds)
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\nTotal simulation time: {int(hours):02}:{int(minutes):02}:{seconds:.2f} (hours:minutes:seconds)")

# Find best model
optimal_param_idx = mse_train_list.index(min(mse_train_list))                   # Replace with the desired iteration index
optimal_model = pinn_param[optimal_param_idx]
r2_max = r2_train_list[optimal_param_idx]
mse_min = mse_train_list[optimal_param_idx]

# Save best model
model_save_path = os.path.join(output_dir, f'model ({T}, {r}).pt')
torch.save(optimal_model, model_save_path)

#%% Testing

# Load a specific set of parameters after training
pinn.load_state_dict(torch.load(model_save_path))

# Make prediction with the best model 
_, H_bed, _, _ = get_rho_eps(T, r)

with torch.no_grad():
    u = pinn(t_plot).detach()

u_pred = torch.clamp(u[:, 0].view(-1, 1), 0.0, H_bed)
P = torch.clamp(u[:, 1].view(-1, 1), 0.0, 1.0)

u_pred_train = pinn(t_obs).detach()[:, 0].view(-1, 1)

# Plot the figure of the axial position
plt.figure(figsize=(6, 5))
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False   
plt.scatter(t_plot, average_pos/H_bed, color='black', label=f'Experiments (T={T}, FI={r})', s=5)  # Scatter plot for the average
plt.fill_between(t, (average_pos - std_dev)/H_bed, (average_pos + std_dev)/H_bed, color='gray', alpha=0.3)  # Error band
plt.plot(t_plot, u_pred/H_bed, label="PINN solution", color="red")
plt.xlabel("t [s]")
plt.ylabel(r"$Z_p$ [-]")
plt.ylim(-0.01, 1.01)
plt.title(f"Best model (Iteration: {optimal_param_idx})")
plt.legend()
# Save the figure in high quality as PNG
file_path = os.path.join(output_dir, f"OptimalDisplac_{optimal_param_idx}.png")
plt.savefig(file_path, dpi=600, bbox_inches='tight')

# Plot the figure of the P parameter
plt.figure(figsize=(6, 5))
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.plot(t_plot, P, color="black")
plt.xlabel("t [s]")
plt.ylabel(r"$P$ [-]")
plt.title(f"Best model (Iteration: {optimal_param_idx})")
x_min, x_max = min(t_plot), max(t_plot)  
y_min, y_max = min(P), max(P)           
x_pos = x_max - (x_max - x_min) * 0.01 -  10
y_pos_top = y_max - (y_max - y_min) * 0.01  
plt.text(x=x_pos, y=y_pos_top-0.008, 
         s=f'$T$ = {T}', fontsize=15, 
         verticalalignment='top', horizontalalignment='right')
plt.text(x=x_pos, y=y_pos_top - (y_max - y_min) * 0.08-0.008,                   # Adjust spacing for stacking
         s=f'$FI$ = {r}', fontsize=15, 
         verticalalignment='top', horizontalalignment='right')
file_path = os.path.join(output_dir, f"OptimalP_param_{optimal_param_idx}.png")
plt.savefig(file_path, dpi=600, bbox_inches='tight') 


# Plot the figure P vs Zexp
plt.figure(figsize=(12, 10))
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.scatter(u_pred/H_bed, P, color="black", alpha=0.5)
plt.xlabel("Z [-]")
plt.ylabel("P [-]")
plt.title(f"Best Model (Iteration: {optimal_param_idx})", fontsize=16)
file_path = os.path.join(output_dir, f"OptimalP_paramVSZ_{optimal_param_idx}.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')  


# Parity plot
plt.figure(figsize=(10, 10))
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.scatter(u_obs, u_pred_train, color="black", alpha=0.5)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title(f"Best model (Iteration: {optimal_param_idx})")
min_val = min(min(u_obs.flatten()), min(u_pred_train.flatten()))
max_val = max(max(u_obs.flatten()), max(u_pred_train.flatten()))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='-', linewidth=2)
x_vals = np.linspace(min_val, max_val, 100)
plt.fill_between(x_vals, x_vals * 0.9, x_vals * 1.1, color='black', alpha=0.2)
plt.text(x=min_val + min_val*0.01, y=max_val - max_val*0.005, 
         s=f'$R^2$ = {r2_max:.3f}', fontsize=15, 
         verticalalignment='top', horizontalalignment='left')
plt.text(x=min_val + min_val*0.01, y=max_val - max_val*0.050,
         s=f'MSE = {mse_min:.3e}', fontsize=15, 
         verticalalignment='top', horizontalalignment='left')
plt.text(x=min_val + min_val*0.01, y=max_val - max_val*0.080,
         s=f'T = {T}', fontsize=15, 
         verticalalignment='top', horizontalalignment='left')
plt.text(x=min_val + min_val*0.01, y=max_val - max_val*0.12,
         s=f'FI = {r}', fontsize=15, 
         verticalalignment='top', horizontalalignment='left')
plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
file_path = os.path.join(output_dir, f"OptimalParity_train{optimal_param_idx}.png")
plt.savefig(file_path, dpi=600, bbox_inches='tight')
