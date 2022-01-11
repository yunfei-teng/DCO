# Codes

We provide the codes used for generating the experimental results in our paper:

## Reproduction

1. To reproduce the experimental results of **Sectoin 3**, please run `python cone_mnist.py` and `python cone_cifar10.py`. The generated figures will show in the folder *illustrative_example* and *illustrative_example_cifar10*.

2. To reproduce the experimental results or check the correctness of our implementation of **Sectoin 5**, please take a look at the bash files `run_mnist.sh`, `run_cifar100.sh` and `run_fashion_mnist.sh` (these codes were tested on a four-gpu machine). The intermediate results can be visualized in Visdom (https://github.com/facebookresearch/visdom) once the bash file starts to be run.

## Implementation details for Section 5

Most implementation details can be found in `main_[dataset].py`. Detailed comments are provided in the codes to help the readers understand the structure and logic of the codes. The usage of each file is listed as follows:

* `main_[dataset].py`: the major file which contains all continual learning methods

* `data.py`: define how we generate and process each continual learning dataset

* `models.py`: define all architectures we use for the experiments

* `trainer.py`: define training and testing functions

* `algorithm.py`: define the algorithm for training the linear autoenocder

* `utils.py`: provide handy tools for other functions

* `options.py`: provide options for hyperparameter search

*The codes were inspired by open source codes [A-GEM](https://github.com/facebookresearch/agem). We made necessary revision and tested the codes on a four-gpu machine.*