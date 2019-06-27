# Bayesian modeling of Spectral data with non-stationary Gaussian processes

Working repository for the special course at DTU.

All relevant codefiles are in the `pytorch_autograd` folder.  

The following scripts and functions are important:

 - `run_inference.py`
   - Main script that performs inference. 
   - Contains the log-probability of the model as well as some useful wrappers
   - Loads data from a `.mat`file and extracts relevant information as well as true solution
   - It runs relevant sampling scheme and saves samples in the `samples_dict` variable. Note that some of the samples are in a transformed domain and should be transformed back to original domain for evaluation
   
  - `nuts.py`
    - Contains implementation of the *No U-turn Sampler* implemented as a class (described in separate repository)
    - Contains simple implementation of the famous Metropolis sampler with similar API
    
  - `aux_funcs_torch.py`
    - Contains auxiliary functions needed - transformations, derivatives of transformations where applicable.
    - Contains implementation of lengthscale function as well as the Gibbs kernel which introduces the non-stationarity to the Gaussian process.
    
   
