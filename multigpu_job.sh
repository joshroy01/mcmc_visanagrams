#!/bin/bash

# The original pipeline isn't setup (to my knowledge) for multi-gpu runs. So instead, we specify via
# command line what GPUs are available per job. Hacky, but boy does it work.
CUDA_VISIBLE_DEVICES=0 conda run -n mcmc_visanagrams doit -f multirun_jobs/dodo.py *seed0
CUDA_VISIBLE_DEVICES=1 conda run -n mcmc_visanagrams doit -f multirun_jobs/dodo.py *seed90210
CUDA_VISIBLE_DEVICES=2 conda run -n mcmc_visanagrams doit -f multirun_jobs/dodo.py *seed8675309