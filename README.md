# world-music-tagging

Automatic tagging for world music recordings using convolutional neural networks. 

## Overview

This project trains a Convolutional Neural Network (CNN) on Mel spectrograms derived from world music recordings of the Smithsonian Folkways and British Library Sound collections. 

![alt tag](https://raw.githubusercontent.com/mpanteli/world-music-tagging/master/data/cnn.png)

## Usage

#### Models

Two CNN architectures are compared against a baseline model trained with MFCCs. 

```python
python scripts/train_model.py
```

See also the notebook [compare_models.ipynb](https://github.com/mpanteli/world-music-tagging/blob/master/notebooks/compare_models.ipynb).
