import torch
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

#Training loop
def train(model, loss_fnc, optimizer, train_loader, val_loader, epochs):
    counter = []
    train_loss = []
    val_loss = []
    images = []

    for epoch in range (epochs):
        loss=0
        count=0
        for i, dat in enumerate(train_loader): 
            (dat_colour, dat_bw) = dat

            optimizer.zero_grad()
            fake_colour = model(dat_bw)
            gen_error = loss_fnc(fake_colour, dat_colour)
            gen_error.backward()

            optimizer.step()
            count=i+1
            loss+=float(gen_error)

        train_loss.append(loss/count)
        vloss=validate(model, val_loader, loss_fnc)
        val_loss.append(vloss)
        counter.append(epoch+1)
        print("epoch:",epoch+1,"training loss: ", f'{loss/count:.4f}',"validation loss: ", f'{vloss:.4f}')
        
    history = [counter, train_loss, val_loss]
    return history

#Plot loss with respect to Epochs
def plot(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history[0], history[1], label='Training Loss')
    plt.plot(history[0], history[2], label = 'Validation Loss')
    plt.legend()
    plt.show()

def inference(loader, model):
    load=next(iter(loader))

    data=load[1]
    label=load[0]
    output=model(data)

    #Undo data transformations
    label=(label*0.5)+0.5
    data=(data*0.5)+0.5
    output=(output*0.5)+0.5

    plt.figure()

    f, axarr = plt.subplots(1,3)

    axarr[0].imshow(data[0].permute(1, 2, 0))
    axarr[1].imshow(output[0].detach().permute(1, 2, 0))
    axarr[2].imshow(label[0].permute(1, 2, 0))
    plt.show()
    
def save(model, PATH = "unet.pt"):
    # Save
    torch.save(model, PATH)

def load(model, PATH = "unet.pt"):
    # Load
    model = torch.load(PATH)
    model.eval()
    return model

def main():
    epochs=10
    (train_loader, val_loader, test_loader)=dataset(batch_size=64)
    model, loss_fncG, optimizerG = initialize_model(lr = 0.0001)
    
    history=train(model, loss_fncG, optimizerG, train_loader, val_loader, epochs)
    plot(history)
    
    print("Training Loss:",validate(model, train_loader, loss_fncG))
    print("Validation Loss:",validate(model, val_loader, loss_fncG))
    print("Test Loss:",validate(model, test_loader, loss_fncG))
    
    inference(train_loader, model)
    inference(val_loader, model)
    
    save(model)
    
    
if __name__ == "__main__":
    main()