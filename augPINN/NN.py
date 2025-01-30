#%% Packages
import sys
sys.path.append(r'C:\Users\Stefano\OneDrive - University College London\Desktop\GitHub Repos\augPINN')  
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from density_bed import get_rho_eps
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
    

#%% Loss definition
def loss_data(model, t, u_exact):
    
    u = model(t)
    
    loss = torch.abs(torch.mean((u - u_exact)))
    
    return loss


#%% Functions to plot and update figures
def live_plotLog(file_path, x_vals, y_vals, title, ylabel, xlabel='Iteration', y_labels= None):
    plt.figure(figsize=(10, 6))
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

t = np.hstack(times).reshape(-1, 1)
z = np.vstack(positions)[:, 0]


#%% Take average of the runs

max_len = max(len(positions[0]), len(positions[1]), len(positions[2]))
t = max(times, key = len)

# Create new arrays with NaNs filled up to max_len
padded_run1 = np.pad(positions[0][:, 0], (0, max_len - len(positions[0])), constant_values=np.nan)
padded_run2 = np.pad(positions[1][:, 0], (0, max_len - len(positions[1])), constant_values=np.nan)
padded_run3 = np.pad(positions[2][:, 0], (0, max_len - len(positions[2])), constant_values=np.nan)

# Take the element-wise mean, ignoring NaNs
average_pos = np.nanmean([padded_run1, padded_run2, padded_run3], axis=0)
std_dev = np.nanstd([padded_run1, padded_run2, padded_run3], axis=0)


#%% Train/test split 80/20

dataset = shuffle(np.hstack((np.hstack(t).reshape(-1, 1), np.vstack(average_pos))), random_state=92)

train = dataset[:int(len(dataset)*0.8), :]
test = dataset[int(len(dataset)*0.8):, :]

#%% Convert to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train
t_obs = torch.tensor(train[:, 0], dtype=torch.float32).view(-1, 1).to(device)
u_obs = torch.tensor(train[:, 1], dtype=torch.float32).view(-1, 1).to(device)

# Test
t_obsTest = torch.tensor(test[:, 0], dtype=torch.float32).view(-1, 1).to(device)
u_obsTest = torch.tensor(test[:, 1], dtype=torch.float32).view(-1, 1).to(device)

# For plotting
t_plot = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)


# Extract the directory from the file path
path = r"C:/Users/Stefano/OneDrive - University College London/Desktop/GitHub Repos/augPINN/data/PP_Pyr/"
output_dir = os.path.join(os.path.dirname(path), 'output_images_500-2_NN')
os.makedirs(output_dir, exist_ok=True)


#%% Training and testing
torch.manual_seed(92)

# Initialize the model and optimizer for each fold
nn = FCN(1, 1, 64, 4).to(device)

# Update the optimizer to include the adaptive weights
optimizer = torch.optim.Adam(nn.parameters(), lr=1e-3)

# Early stopping parameters (if needed)
# patience = 200                                                                  # Early stopping patience
# tolerance = 1e-4                                                                # Tolerance for early stopping
# best_val_loss = float('inf')
# early_stopping_counter = 0

# Lists to record useful parameters
epoch_list = []
loss_train_list = []
loss_val_list = []
loss_test_list = []
r2_train_list = [] 
r2_val_list = [] 
r2_test_list = []
mse_train_list = [] 
mse_val_list = [] 
mse_test_list = []
nn_param = []
    

iterations = 10000                                                              # Number of iterations for training

metric = R2Score()


# Scalar inputs for the physics loss
T = 500
r = 2

start_time = time.time()                                                        # Capture the start time


# Training loop
for i in range(iterations):
    optimizer.zero_grad()
        
    loss_train = loss_data(nn, t_obs, u_obs)
        
    # Backpropagation and optimization step
    loss_train.backward()
    optimizer.step()
    
    # Save the current parameters
    nn_param.append(copy.deepcopy(nn.state_dict()))
    
        
    # Test phase
    loss_test = loss_data(nn, t_obsTest, u_obsTest)
        
        
    # Record losses
    epoch_list.append(i + 1)
    loss_train_list.append(loss_train.item())
    loss_test_list.append(loss_test.item())
    
    # Make predictions
    with torch.no_grad():
        u_pred_train = nn(t_obs).detach()[:, 0].view(-1, 1)
        u_pred_test = nn(t_obsTest).detach()[:, 0].view(-1, 1)
    r2_train_list.append(metric(u_pred_train, u_obs).item())
    r2_test_list.append(metric(u_pred_test, u_obsTest).item())
    mse_train_list.append(torch.mean((u_pred_train - u_obs)**2).item())
    mse_test_list.append(torch.mean((u_pred_test - u_obsTest)**2).item())
        
        
        
    # # Early stopping check (if needed)
    # if loss_total_val < best_val_loss - tolerance:
    #     best_val_loss = loss_total_val
    #     early_stopping_counter = 0
    # else:
    #     early_stopping_counter += 1

    # if early_stopping_counter >= patience:
    #     print(f"Early stopping at epoch {epoch + 1}")
    #     break
        

    # Print and plot every 100 iterations
    if i % 100 == 0:
        print(f'Iteration {i}/{iterations}', 
                  f'Loss train: {loss_train.item():.4f}',
                  f'Loss test: {loss_test.item():.4f}')
            
        # Make overal predictions for plotting
        _, H_bed, _, _ = get_rho_eps(T, r)
        with torch.no_grad():
            u = nn(t_plot).detach()
        u_pred = torch.clamp(u[:, 0].view(-1, 1), 0.0, H_bed)
            
        # Plot the loss curves and parameters
        file_path = os.path.join(output_dir, "Losses.png")
        live_plotLog(file_path, epoch_list, [loss_train_list, loss_test_list],
                         title='Losses', ylabel='Loss', 
                         y_labels=['Train', 'Test'])
            
            
        file_path = os.path.join(output_dir, "MSE.png")
        live_plotLog(file_path, epoch_list, [mse_train_list, mse_test_list], 
                         title='MSE', 
                         ylabel='MSE', y_labels=['Train', 'Test'])
            
            
        # Plot the figure displac
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
optimal_model = nn_param[optimal_param_idx]
r2_max = r2_train_list[optimal_param_idx]
mse_min = mse_train_list[optimal_param_idx]

# Save best model
model_save_path = os.path.join(output_dir, f'model ({T}, {r}).pt')
torch.save(optimal_model, model_save_path)

#%% Testing

# Load a specific set of parameters after training
nn.load_state_dict(torch.load(model_save_path))

# Make prediction with the best model 
_, H_bed, _, _ = get_rho_eps(T, r)

with torch.no_grad():
    u = nn(t_plot).detach()

u_pred = torch.clamp(u[:, 0].view(-1, 1), 0.0, H_bed)

u_pred_train = nn(t_obs).detach()[:, 0].view(-1, 1)

# Plot the figure displac
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
plt.savefig(file_path, dpi=600, bbox_inches='tight')  # dpi=600 for high quality

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
