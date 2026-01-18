# Computational Plasticity in Python

A Python library for the numerical implementation of constitutive models for small-strain plasticity. This repository focuses on the algorithmic implementation of return-mapping algorithms for rate-independent and rate-dependent plasticity using fully implicit backward-Euler integration.

# Features
## * J2 Flow Theory (Von Mises):

  ** Radial return mapping algorithm.
  
  ** Non-linear isotropic hardening (Voce/Saturation type).
  
  ** Linear kinematic hardening (Prager).
  
  ** Newton-Raphson solver for the consistency parameter $\Delta \gamma$.

## * 1D Plasticity:

  ** Uniaxial stress-strain implementation for algorithmic validation.
  
  ** Viscoplasticity (Perzyna regularization).
  
  ** Coupled isotropic/kinematic hardening.
  
# Mathematical Formulation

The core solver implements the return mapping algorithm by solving the consistency condition $f(\sigma_{n+1}, \alpha_{n+1}) = 0$. For non-linear hardening rules, this results in a non-linear scalar equation for the discrete plastic multiplier $\Delta \gamma$, which is solved via a local Newton-Raphson loop:

$$\Delta \gamma^{(k+1)} = \Delta \gamma^{(k)} - \frac{g(\Delta \gamma^{(k)})}{g'(\Delta \gamma^{(k)})}$$

where $g(\Delta \gamma)$ is the residual of the yield function.

# Structure

  * src/plasticity.py: Contains the core Newton-Raphson solvers (solve_1d_newton_raphson and solve_j2_newton_raphson).
  * examples/: Contains driver scripts to run simulations and plot results.
    ** run_1d_simulation.py: Simulates a uniaxial loading-unloading cycle.
    ** run_j2_simulation.py: Simulates a 3D stress state under cyclic loading.

# Usage
To run the examples, clone the repository and execute the scripts from the root directory:
python examples/run_1d_simulation.py
python examples/run_j2_simulation.py


# Author
Mariano Tomás Fernandez - Implementation and development

