import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

# Parent directory to path (import src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plasticity import solve_j2_newton_raphson

def dev(tensor_voigt):
    """
    Computes deviatoric part of a 6x1 Voigt tensor.
    tensor = [xx, yy, zz, xy, xz, yz]
    """
    trace = np.sum(tensor_voigt[0:3])
    volumetric = (trace / 3.0) * np.array([1, 1, 1, 0, 0, 0])
    return tensor_voigt - volumetric

def run_j2_loading():
    print("Running J2 Plasticity Simulation...")

    # ___________________________________________________________________________#
    # GENERAL INPUT PARAMETERS

    # Material parameters
    E = 9e10  # [Pa] - Steel Young modulus
    v = 0.3   # [-] - Poisson ratio
    s_Y = 4.2e8  # [Pa] - Yield stress
    
    # Hardening parameters
    K = 0  # Isotropic Linear
    H = 0  # Kinematic Linear
    
    # Rate dependent parameters
    nu = 0  # [Pa*s] - Viscosity (0 = rate independent)
    t_tot = 5.0  # [s]
    Dt = 0.025  # [s]
    
    # Non-linear Isotropic Hardening:
    NL = 0  # [-]
    sigma_inf = 9.0e8
    delta_NL = 150

    # Derived Elastic Constants
    mu = E / (2 * (1 + v))
    
    # ___________________________________________________________________________#
    # STRAIN INPUT PARAMETERS
    
    # Loading Path: 0.02 -> 0 -> -0.02 -> 0 -> 0.02 (Axial XX)
    # The original code defined a specific matrix of loading cases
    loading_cases = np.array([
        [0.02, 0.00, -0.02, 0.0, 0.02], # XX
        [0.0, 0.0, 0.0, 0.0, 0.0],      # YY
        [0.0, 0.0, 0.0, 0.0, 0.0],      # ZZ
        [0.0, 0.0, 0.0, 0.0, 0.0],      # XY
        [0.0, 0.0, 0.0, 0.0, 0.0],      # XZ
        [0.0, 0.0, 0.0, 0.0, 0.0]       # YZ
    ])
    
    load_cases_count = loading_cases.shape[1]
    steps_per_case = int(t_tot / Dt)
    total_steps = load_cases_count * steps_per_case
    
    strain_mat = np.zeros((total_steps + 1, 6))
    
    # Generate strain path
    current_strain = np.zeros(6)
    idx = 0
    prev_target = np.zeros(6)
    
    for j in range(load_cases_count):
        target = loading_cases[:, j]
        d_eps = (target - prev_target) / steps_per_case
        
        for i in range(steps_per_case):
            current_strain += d_eps
            idx += 1
            if idx <= total_steps:
                strain_mat[idx, :] = current_strain
        prev_target = target

    # ___________________________________________________________________________#
    # MAIN LOOP
    
    # State Variables Storage
    # eps_p_mat stores 13 components:
    #   [0:6]  -> Plastic Strain Tensor (eps_p)
    #   [6]    -> Accumulated Plastic Strain (chi)
    #   [7:13] -> Kinematic Hardening Strain (eps_p_kin)
    eps_p_mat = np.zeros((len(strain_mat), 13))
    # Stress Storage
    # stress_mat stores 13 components:
    #   [0:6]  -> Stress Tensor (sigma)
    #   [6]    -> Isotropic Hardening Stress (R)
    #   [7:13] -> Back Stress Tensor (q)
    stress_mat = np.zeros((len(strain_mat), 13))    
    # Deviatoric stress for plotting
    devstress_mat = np.zeros((len(strain_mat), 6))
    
    for j in range(1, len(strain_mat)):
        # Retrieve state from previous step (n)
        eps_p_n = eps_p_mat[j-1, 0:6]
        chi_n = eps_p_mat[j-1, 6]
        eps_p_kin_n = eps_p_mat[j-1, 7:13]
        
        # Current total strain input
        strain = strain_mat[j, :]
        
        # --- Elastic Predictor (Trial State) ---
        
        # 1. Deviatoric Strain Calculation
        trace_eps = np.sum(strain[0:3])
        vol_eps = trace_eps / 3.0
        dev_eps = strain - vol_eps * np.array([1, 1, 1, 0, 0, 0])
        
        # 2. Trial Deviatoric Stress
        # s_trial = 2 * mu * (e_dev - e_dev_p)
        # Using eps_p_n as the plastic deviatoric strain (assuming isochoric)
        s_trial = 2 * mu * (dev_eps - eps_p_n)
        
        # 3. Trial Back Stress (Deviatoric)
        # q = (2/3) * H * eps_p_kin
        alpha_dev = (2/3) * H * eps_p_kin_n
        
        # 4. Relative Stress (xi)
        xi_trial = s_trial - alpha_dev
        norm_xi = np.linalg.norm(xi_trial)
        
        # 5. Trial Isotropic Hardening (R)
        if NL == 0:
            R_trial = K * chi_n
        else:
            R_trial = (sigma_inf - s_Y) * (1 - math.exp(-delta_NL * chi_n)) + K * chi_n
            
        # 6. Yield Function Evaluation
        # f = ||xi|| - sqrt(2/3) * (s_Y + R)
        sqrt_23 = (2/3)**0.5
        f_trial = norm_xi - sqrt_23 * (s_Y + R_trial)
        
        if f_trial <= 0:
            # --- Elastic Step ---
            s_updated = s_trial
            # Reconstruct total stress: sigma = s + K_bulk * trace(eps) * I
            # Note: 3*K_bulk = E / (1-2v)
            sigma_updated = s_updated + (E / (3*(1-2*v)) * trace_eps) * np.array([1, 1, 1, 0, 0, 0])
            
            # Update storage (state remains unchanged)
            stress_mat[j, 0:6] = sigma_updated
            stress_mat[j, 6] = R_trial
            stress_mat[j, 7:13] = alpha_dev # Store back stress
            eps_p_mat[j, :] = eps_p_mat[j-1, :]
            
        else:
            # --- Plastic Corrector ---
            if NL == 0:
                # Analytical solution for linear hardening
                denom = 2*mu + (2/3)*K + (2/3)*H + (nu/Dt)
                gamma = f_trial / denom
            else:
                # Solve non-linear equation for gamma
                gamma = solve_j2_newton_raphson(
                    chi_n, f_trial, Dt, mu, H, K, nu, sigma_inf, s_Y, delta_NL
                )
            
            # Flow direction
            if norm_xi > 1e-12:
                n_flow = xi_trial / norm_xi
            else:
                n_flow = np.zeros(6)
            
            # Update Plastic State Variables
            # Plastic strain: eps_p = eps_p_n + gamma * dt * n
            eps_p_mat[j, 0:6] = eps_p_n + gamma * Dt * n_flow
            
            # Accumulated plastic strain: chi = chi_n + gamma * dt * sqrt(2/3)
            eps_p_mat[j, 6] = chi_n + gamma * Dt * sqrt_23
            
            # Kinematic plastic strain
            eps_p_mat[j, 7:13] = eps_p_kin_n + gamma * Dt * n_flow 
            
            # Update Stresses (Radial Return)
            # s = s_trial - 2*mu*gamma*dt*n
            s_updated = s_trial - 2 * mu * gamma * Dt * n_flow
            
            # Total Stress
            sigma_updated = s_updated + (E / (3*(1-2*v)) * trace_eps) * np.array([1, 1, 1, 0, 0, 0])
            
            stress_mat[j, 0:6] = sigma_updated
            
            # Update Hardening Stress R
            if NL == 0:
                stress_mat[j, 6] = K * eps_p_mat[j, 6]
            else:
                stress_mat[j, 6] = (sigma_inf - s_Y) * (1 - math.exp(-delta_NL * eps_p_mat[j, 6])) + K * eps_p_mat[j, 6]
                
            # Update Back Stress q
            stress_mat[j, 7:13] = (2/3) * H * eps_p_mat[j, 7:13]
                
        # Store deviatoric stress for plotting
        devstress_mat[j, :] = dev(stress_mat[j, 0:6])

    # 4. Plot Results
    strain_xx = strain_mat[:, 0]
    stress_xx = stress_mat[:, 0] / 1e6 # Convert to MPa
    dev_stress_xx = devstress_mat[:, 0] / 1e6
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Axial Stress vs Axial Strain
    plt.subplot(1, 2, 1)
    plt.plot(strain_xx, stress_xx, '--k', label=r"$\sigma_{xx}$ [J2]")
    plt.grid(True)
    plt.xlabel(r'Axial strain, $\epsilon_{xx}$ [-]')
    plt.ylabel(r'Axial stress, $\sigma_{xx}$ [MPa]')
    plt.title('Axial Stress-Strain Response')
    plt.legend()
    
    # Plot 2: Deviatoric Stress vs Strain
    plt.subplot(1, 2, 2)
    plt.plot(strain_xx, dev_stress_xx, '-b', label=r"dev($\sigma$)$_{xx}$")
    plt.grid(True)
    plt.xlabel(r'Axial strain, $\epsilon_{xx}$ [-]')
    plt.ylabel(r'Deviatoric Stress [MPa]')
    plt.title('Deviatoric Stress Response')
    
    plt.tight_layout()
    print("J2 Simulation complete.")
    plt.show()

if __name__ == "__main__":
    run_j2_loading()