import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plasticity import solve_1d_newton_raphson

def trial_1d_check(C, eps_p, chi, chi_r, strain, s_Y, NL, sigma_inf, delta):
    """
    Computes trial state for 1D Plasticity.
    Based on 'Trial' function from original code.
    """
    E = C[0,0]
    K = C[1,1]
    H = C[2,2]
    
    # Elastic Predictor
    # stress = E * (strain - plastic_strain)
    sigma_trial_stress = E * (strain - eps_p)
    
    # Back stress (Kinematic)
    # alpha_kin = H * plastic_strain
    # sigma_trial[0] = stress
    # sigma_trial[1] = R (Isotropic Hardening Stress)
    # sigma_trial[2] = q (Back Stress)
    
    # Isotropic Hardening Force R
    if NL == 0:
        R_trial = K * chi
    else:
        R_trial = (sigma_inf - s_Y) * (1 - np.exp(-delta * chi)) + K * chi
        
    # Back Stress q
    q_back_trial = H * eps_p 
    
    # Yield Function
    # f = |sigma - q| - (s_Y + R)
    xi_trial = sigma_trial_stress - q_back_trial
    f_trial = abs(xi_trial) - (s_Y + R_trial)
    
    return f_trial, sigma_trial_stress, R_trial, q_back_trial


def run_1d_loading():
    print("Running 1D Plasticity Simulation...")

    # 1. Parameters (from your original code)
    E = 2.1e11 
    s_Y = 4.2e8 
    
    # Hardening
    K = 2e10 # Isotropic
    H = 1e10 # Kinematic
    C = np.diag([E, K, H])
    
    # Viscosity
    nu = 3e10
    t_tot = 100.0
    dt = 0.025
    
    # Non-linear hardening
    NL = 1 # 1 = On
    sigma_inf = 9.0e8
    delta = 25
    
    # 2. Loading Path
    # 0 -> 0.02 -> -0.02 -> 0.02
    loading_cases = np.array([0.02, -0.02, 0.02])
    # Reconstruct the time stepping from original code logic
    total_steps = int(len(loading_cases) * t_tot / dt)
    strain_history = np.zeros(total_steps + 1)
    
    current_strain = 0.0
    steps_per_leg = int(t_tot / dt)
    
    # Generate strain path
    idx = 0
    start_strain = 0.0
    for target_strain in loading_cases:
        d_eps = (target_strain - start_strain) / steps_per_leg
        for i in range(steps_per_leg):
            idx += 1
            current_strain += d_eps
            strain_history[idx] = current_strain
        start_strain = target_strain
        
    # 3. Time Loop
    stress_results = np.zeros_like(strain_history)
    eps_p_results = np.zeros_like(strain_history)
    
    # State variables
    eps_p_n = 0.0 # Plastic strain
    chi_n = 0.0   # Accumulated plastic strain (isotropic)
    
    for i in range(1, len(strain_history)):
        strain = strain_history[i]
        
        # Trial
        # Compute trial stress assuming elastic step
        sigma_trial = E * (strain - eps_p_n)
        
        # Compute back stress
        q_back = H * eps_p_n
        
        # Compute Isotropic Hardening R
        if NL == 1:
            R_n = (sigma_inf - s_Y) * (1 - np.exp(-delta * chi_n)) + K * chi_n
        else:
            R_n = K * chi_n
            
        # Yield function
        xi_trial = sigma_trial - q_back
        f_trial = abs(xi_trial) - (s_Y + R_n)
        
        if f_trial <= 0:
            # Elastic Step
            sigma_updated = sigma_trial
        else:
            # Plastic Step
            gamma = solve_1d_newton_raphson(
                    C, f_trial, nu, dt, sigma_inf, s_Y, delta, chi_n)
            
            # Update
            sgn = np.sign(xi_trial)
            
            # Update Plastic Strain
            eps_p_n += gamma * dt * sgn
            chi_n += gamma * dt
            
            # Update Stress
            # sigma = sigma_trial - gamma*dt*E*sgn
            sigma_updated = sigma_trial - gamma * dt * E * sgn
            
        stress_results[i] = sigma_updated
        eps_p_results[i] = eps_p_n

    # 4. Plot
    time_axis = np.linspace(0, len(strain_history)*dt, len(strain_history))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(strain_history, stress_results/1e6, 'k', linewidth=1.5, label='Hysteretic Loop')
    plt.xlabel('Strain [-]')
    plt.ylabel('Stress [MPa]')
    plt.title('Stress-Strain Response')
    plt.grid(True) # Fixed the 'b=True' deprecation warning
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(time_axis, stress_results/1e6, 'b', label='Stress History')
    plt.xlabel('Time [s]')
    plt.ylabel('Stress [MPa]')
    plt.title('Time History')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_1d_loading()