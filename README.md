# Cherry tree skeletionization

This project aims to improve on the method described in the paper '[Semantics-guided Skeletonization of Sweet Cherry Trees for
Robotic Pruning](https://arxiv.org/pdf/2103.02833.pdf)'. The datasets used for this paper and this project can be found [here](https://paperswithcode.com/dataset/ufo-cherry-tree-point-clouds)

## Setup
> **Important**: some libraries used in this project are not yet published for python 3.11, therefore python 3.10 must be used

To install the required packages run the following command:
```shell
pip install -r requirements.txt
```

A config file must be created in `Cherry-trees/config` named `config.ini`. In this config file the data directory is defined. The format is given below.

```ini
[DATA]
PATH = /path/to/data/directory/
```

## Usage
The project can be run using different strategies. The three different strategies can be run as follows:

Method described by the paper using a CNN:
```shell
python Cherry-trees/src/main.py --method "cnn" --bag <bag_number>
```

Persistent homology for edge confidence. The replacement for the CNN
```shell
python Cherry-trees/src/main.py --method "homology" --bag <bag_number>
```

New method using a reebgraph
```shell
python Cherry-trees/src/main.py --method "reeb" --bag <bag_number>
```
