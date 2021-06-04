# GRMF

This repository is the official implementation of [Robust Matrix Factorization with Grouping Effect](). 

## Requirements
To install Python package from github, you need to clone a repository.
```
git clone https://github.com/GroupingEffects/GRMF
```

Then just run the `setup.py` file

```
sudo python setup.py install
```


## Dependencies
GRMF requires:
* Python (>=3.8)
* NumPy (>=1.14.6)
* Pandas (==0.25.3)
* sklearn
* glob
* PIL
* scipy
* multiprocessing
* abc
* time
* warning

## Description
* utils.py: Base functions for evaluation， function adding corruption to the data
* data_loader.py: Load the pictures into a numpy array form
* benchmark_algorithm.py: Contain all the benchmark algorithms classes or functions
* settings.py: The dictionary of the hyper-parameters of the GRMF
* GRMF.py: The GRMF class
* main.py: Run the main experiment in the paper


## Run

```train
python main.py
```
Or run the main() in the `main.py` file using some IDEs (e.g. Pycharm).


