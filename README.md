
## Installation 
Install the environment first
```bash
conda create -n LFPfit_env python=3.11  conda-forge::pytorch=2.0.0 numpy=1.24 matplotlib pandas scipy 
```
After succesfully installed the environment, 
```bash
conda activate LFPfit_env
```
and then install the package
```bash
pip install .
```