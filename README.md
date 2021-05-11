
## File Contents
This folder contains the source code which we used for our Final Project. This code is found in the following to notebook files
- Make_Dataset_Folders.ipynb: For creating the directory structure and image folders needed for the Main.ipynb notebook

- Main.ipynb: For testing and evaluating various AlexNet, ResNet-18, ResNet-50, and DenseNet-121 models on the train, validation, and test folders.


##Usage Instructions

Make_Dataset_Folders.ipynb should be executed in its entirety before running Main.ipynb. The purpose of this notebook is to create the directory structure and image folders needed in 
the Main.ipynb notebook. This notebook also expects the contents of the Kaggle Dataset to be manually downloaded into the "./root/data/Kaggle Dataset" path.

Once Make_Dataset_Folders.ipynb is executed, Main.ipynb should be ready to run. 

## Toolkits Used

Special thanks go out to the developers of ImbalancedDatasetSampler which is provided by the following github repo https://github.com/ufoym/imbalanced-dataset-sampler. 
This sampler is designed to be used with PyTorch dataloaders, and implements a balance in class distribution for sampled data to the dataloader. This package was installed in Main.ipynb with the following command:

```python 
!pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
from torchsampler import ImbalancedDatasetSampler
```

The following packages are also required for execution of our notebooks

```python
import os
import pickle  
import random
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time; _START_RUNTIME = time.time()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pandas as pd
```
