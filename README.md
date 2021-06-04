# GRMF



This repository is the official implementation of **Robust Matrix Factorization with Grouping Effect**. 
## Requirements
To install Python package from github, you need to clone a repository.
```
git clone https://github.com/GroupingEffects/GRMF
```
Then just run the setup.py file

```
sudo python setup.py install
```
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

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
Or run the main() in the main.py file using some IDEs (e.g. Pycharm).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
