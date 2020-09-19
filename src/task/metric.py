from pytorch_lightning.metrics import functional as FM
import torch.nn as nn

cross_entropy = nn.CrossEntropyLoss()
acc = FM.accuracy
