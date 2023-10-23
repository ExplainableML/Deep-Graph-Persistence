# Deep Graph Persistence
This is the code for our work [Addressing caveats of neural persistence with deep graph persistence](https://arxiv.org/abs/2307.10865).

## Overview
This repository contains code for replicating our empirical experiments regarding neural persistence (NP) in trained neural networks.
Also, this repository contains code for replicating our experiments on using deep graph persistence for detecting image corruptions.

For calculating NP, we rely on the [great code](https://github.com/BorgwardtLab/Neural-Persistence) from the original [Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology](https://arxiv.org/abs/1812.09764) paper.

## Setup
Besides standard dependencies available via `pip`, this repository also requires the `Aleph` library.
Detailed instructions on how to install `Aleph` can be found in the [corresponding repository](https://github.com/Pseudomanifold/Aleph).

After installing `Aleph`, the remaining dependencies can be installed from [requirements.txt](requirements.txt) via 
```
pip install -r requirement.txt
```

## Replication
### Analysis of Neural Persistence
TODO.

### Detecting corrupted images with Deep Graph Persistence
To follow the instructions, first change to `src/dgp-shift-detection`. We also provide example scripts in `src/dgp-shift-detection/example-scripts` to illustrate running experiments in parallel using the `slurm` resource manager.

Replicating this experiment involves several steps.
The first step is to train MLP models required to calculate metrics such as DGP or Topological Uncertainty.
To train MLP models, run
```bash
python train.py
```
while setting the desired parameters. The most important parameters are:
  * `--hidden`: The hidden size
  * `--layers`: The number of hidden layers
  * `--dataset`: The dataset to train the MLP on. Options are `mnist`, `fashion-mnist`, `cifar10` as defined in [load_data.py](src/dgp-shift-detection/load_data.py)
  * `--runs`: How many replications with the same hyperparameters, but different initialization, we want to train
  * `--checkpoint_root_dir`: Where to save model checkpoints

A full list can be found by running `python train.py --help`.

The next step is preparing the data, i.e. the corrupted images.
To prepare the data, run
```bash
python make_shifted_data.py
```
while setting the desired parameters. The relevant parameters are:
  * `--dataset`: The dataset to take images from. Options are `mnist`, `fashion-mnist`, `cifar10` as defined in [load_data.py](src/dgp-shift-detection/load_data.py)
  * `--num-train-samples`: The number of images to use for training purposes. Recall that some methods, like Topological Uncertainty, require a number of image representations as reference
  * `--num-test-samples`: The number of samples to corrupt for testing purposes

Then, we have to extract the representations of corrupted and non-corrupted (clean) images in various ways.
To extract representations, run
```bash
python make_witness_vectors.py
```
while setting the appropriate parameters. The following parameters are required:
  * `--dataset`: The path to a dataset produced by `make_shifted_data.py` (has file ending `.npy`)
  * `--model-name`: Specifies the model's hyperparameters and must be of form `data=$dataset-hidden=$hidden-layers=$layers-run=$run`
  * `--model-path`: Path to a model checkpoint resulting from running `train.py`. Image representations are extracted from this model
  * `--shift`: The corruption to extract representations for. Options are `gaussian_noise`, `salt_pepper_noise`, `gaussian_blur`, `image_transform`, `uniform_noise`, `pixel_shuffle`, `pixel_dropout`, as defined in [data_shifts.py](src/dgp-shift-detection/data_shifts.py)
  * `--intensity`: The corruption intensity. Must be an integer between (including) 0 and 5
  * `--method`: The method used for extracting image representations. Options are: `softmax`, `logit`, `magdiff`, `dgp`, `dgp_normalize`, `dgp_var`, `dgp_var_normalize`, `dgp_approximate`, `dgp_var_approximate`, `topological`, `topological_var`, `input_activation` as defined in [witness_functions.py](src/dgp-shift-detection/witness_functions.py)

Finally, we evaluate the accuracy of detecting corrupted image (batches) by running
```bash
python magdiff_evaluation.py
```
where the following parameters are required:
  * `--dataset`
  * `--hidden`
  * `--layers`
  * `--run`
  * `--shift`
  * `--intensity`
  * `--method`

These parameters specify the MLP model and the type of corruption for which representations have been extracted by running `make_witness_vectors.py`.
Results will be saved in a subdirectory of `./evaluation_results`.

## References
TODO.
