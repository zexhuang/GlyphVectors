# Learning Transformation-Invariant Geometric Features for Vector Data Classification

## Overview

This is the official code repository of paper *Learning Transformation-Invariant Geometric Features for Vector Data Classification*. Currently, it is submitted to [ACM SIGSPATIAL 2022](https://sigspatial2022.sigspatial.org/) and under review.

## Requirements

* python=3.8
* pytorch=1.10
* shapely
* numpy
* pandas
* freetype
* sklearn

## File tree

```text
GlyphVectors
    |- config
    |    |- config_loader.py      # helper function to load config.yaml.
    |    |- config.yaml           # configs and hyperparameters of models.
    |- data
    |    |- glyph
    |    |   |- ...               # dataset glyph geometry.
    |    |- ttfs
    |    |   |- ...               # list of ttfs to generate glyph polygons.
    |    |- data_generator.py     # dataset class that generates glyph geometries from ttfs.
    |    |- torch_dataset.py      # torch dataset class for glyph geometry. 
    |    |- transform.py          # troch transform functions for data pre-processing.
    |- experiment
    |    |- figs                  # figures used in the paper. 
    |    |   |- ...
    |    |- eval.ipynb            # experiments and results. 
    |- model
    |    |- net.py                # model implementations in pytorch. 
    |- saved                      # saved model checkpoints. 
    |    |- cnn
    |    |   |- ...
    |    |- deepset
    |    |   |- ...
    |    |- gcnn
    |    |   |- ...
    |    |- transformer
    |    |   |- ...
    |- utils
    |    |- metric.py            # metric class that computes avg loss, accuracy and confusion matrix. 
    |    |- pytorchtools.py      # utility function for early-stopping.
    |- trainval.py               # training/validation script. 

```

## Dataset

To generate glyph geometry datasets, you can run

```python
python data/data_generator.py
```

The script would generate two datasets, dataset *transforms* and dataset *similar*, which both include augmented samples, and save to path data/glyph.

## Train/valid implementation

Once datasets are generated, you can train models by setting

```yaml
# Dataset
train_set: [name of training set].pkl
val_set: [name of validation set].pkl
# Training config
train: True
load_state: False
# Model
model: [name of network]
...
```

in config.yaml, and then simply run

```python
python trainval.py
```

To load the saved model checkpoint, in config.yaml you may change

```yaml
# Dataset
val_set: [name of validation set].pkl
# Training config
train: False
load_state: True
save: saved/
checkpoint: [name of checkpoint].pth
# Model
model: [name of network]
...
```

and then simply run

```python
python trainval.py
```

## Citation

```text
Zexian Huang, Kourosh Khoshelham, and Martin Tomko. 2022. Learning Transformation-Invariant Geometric Features for Vector Data Classification. In Proceedings of The ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems 2022 (ACM SIGSPATIAL 2022).
```

## License

MIT License
