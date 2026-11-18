# BACON
Bayesian Active Clustering with Oracle Noise

# Installing
```
# clone the repository
git clone https://github.com/edahelsinki/BACON.git
cd BACON

# set up a conda enviroment
conda create -n bacon python=3.13
conda activate bacon

# install packages
pip install numpy numba scipy scikit-learn matplotlib ucimlrepo
# install clustpy (optional, for experiments only)
pip install git+https://github.com/collinleiber/ClustPy.git 
# install the bacon project package
pip install -e .
```

# Running experiments

Reproducing the comparisons using 10% oracle noise, use the following (the results will be saved in directory `BACON/results`).
```
python experiment.py --experiment compare -o results --epsilon 0.1 --budget 100 --repeats 10
```
For other experiments change `--experiment` value (see documentation in `experiments.py`).