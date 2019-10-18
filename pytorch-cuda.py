import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    print("torch cuda is available")
    print("Numbers of GPU: ", torch.cuda.device_count())
else:
    print("torch cuda is not available")
