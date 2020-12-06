import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import *
from dataset import dataset

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

    
#Validation
def validate(model, val_loader, loss_fnc):
    vloss=0
    count=0
    for i, data in enumerate(val_loader, 0):
            labels, inputs = data
            pred = model(inputs)
            vloss += loss_fnc(pred, labels)
            count = i+1
    
    return float(vloss/count)

def main():
    (train_loader, val_loader, test_loader)=dataset(batch_size=64)
    model, loss_fnc = initialize_baseline()
    
    print("Training Loss:",validate(model, train_loader, loss_fnc))
    print("Validation Loss:",validate(model, val_loader, loss_fnc))
    print("Test Loss:",validate(model, test_loader, loss_fnc))
    
    
if __name__ == "__main__":
    main()