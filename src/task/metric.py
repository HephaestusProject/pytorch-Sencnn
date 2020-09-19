import torch.nn as nn
from pytorch_lightning.metrics import functional as FM

cross_entropy = nn.CrossEntropyLoss()
acc = FM.accuracy
