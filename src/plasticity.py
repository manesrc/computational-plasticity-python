import numpy as np
import math

def solve_1d_newton_raphson(C, f_trial, nu, Dt, sigma_inf, s_Y, delta, chi):
    # Newton - Raphson method for solving non - linear problems
    # intializing the parameters
    k = 0
    gamma_k = 0
    
    eps = 1e-6
    max_it = 20
    
    # calculating the first value of the residual
    # Original logic from your snippet:
    # nonlin = ( sigma_inf - s_Y ) *( math . exp (- delta * chi ) - math . exp (- delta *( chi + gamma_k * Dt ))+C [1 ,1]* gamma_k * Dt )
    nonlin = (sigma_inf - s_Y) * (math.exp(-delta * chi) - math.exp(-delta * (chi + gamma_k * Dt))) + C[1,1] * gamma_k * Dt
    
    # g_k = f_trial - gamma_k * Dt *( C [0 ,0]+ C [2 ,2]+ nu / Dt ) - nonlin
    g_k = f_trial - gamma_k * Dt * (C[0,0] + C[2,2] + nu / Dt) - nonlin
    
    while (abs(g_k) > eps) and (k < max_it):
        # Derivative calculation inferred to match logic
        term_exp = math.exp(-delta * (chi + gamma_k * Dt))
        d_nonlin_dgamma = (sigma_inf - s_Y) * delta * term_exp * Dt + C[1,1] * Dt
        
        Dg_k = -Dt * (C[0,0] + C[2,2] + nu / Dt) - d_nonlin_dgamma
        
        var_gamma = -g_k / Dg_k
        gamma_k = gamma_k + var_gamma
        k = k + 1
        
        # re-calc of the residual (g_k)
        nonlin = (sigma_inf - s_Y) * (math.exp(-delta * chi) - math.exp(-delta * (chi + gamma_k * Dt))) + C[1,1] * gamma_k * Dt
        g_k = f_trial - gamma_k * Dt * (C[0,0] + C[2,2] + nu / Dt) - nonlin
        
    return gamma_k

def solve_j2_newton_raphson(chi, f_trial, Dt, mu, H, K, nu, sigma_inf, s_Y, delta_NL):
    # Newton - Raphson method for solving non - linear problems
    # intializing the parameters
    k = 0
    gamma_k = 0
    
    eps = 1e-6
    max_it = 30
    
    # calculating the first value of the residual
    PI_n1 = (sigma_inf - s_Y) * (1 - math.exp(-delta_NL * (chi + gamma_k * Dt * ((2/3)**0.5)))) + K * (chi + gamma_k * Dt * ((2/3)**0.5))
    PI_n = (sigma_inf - s_Y) * (1 - math.exp(-delta_NL * chi)) + K * chi
    nonlin = ((2/3)**0.5) * (PI_n1 - PI_n)
    g_k = f_trial - gamma_k * Dt * (2 * mu + (2/3) * H + (nu / Dt)) - nonlin
    
    while (abs(g_k) > eps) and (k < max_it):
        deriv2 = (sigma_inf - s_Y) * delta_NL * math.exp(-delta_NL * (chi + gamma_k * Dt * ((2/3)**0.5))) + K
        Dg_k = -(2 * mu + deriv2 + (2/3) * H + (nu / Dt))
        
        var_gamma = -g_k / Dg_k
        gamma_k = gamma_k + var_gamma
        k = k + 1
        
        # re-calc of the residual (g_k)
        PI_n1 = (sigma_inf - s_Y) * (1 - math.exp(-delta_NL * (chi + gamma_k * Dt * ((2/3)**0.5)))) + K * (chi + gamma_k * Dt * ((2/3)**0.5))
        PI_n = (sigma_inf - s_Y) * (1 - math.exp(-delta_NL * chi)) + K * chi
        nonlin = ((2/3)**0.5) * (PI_n1 - PI_n)
        g_k = f_trial - gamma_k * Dt * (2 * mu + (2/3) * H + (nu / Dt)) - nonlin
        
    return gamma_k