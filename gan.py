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
def correctCount(predictions, label):
    correct=0
    
    for i, pred in enumerate(predictions.flatten()):
        if((label[i].item()>=0.5 and pred.item()>=0.5) or (label[i].item()<0.5 and pred.item()<0.5)): #Correct Prediction
            correct+=1
    return(correct)

def train(Generator, loss_fncG, optimizerG, Discriminator, loss_fncD, optimizerD, train_loader, val_loader, epochs):
    counter = []
    Gen_loss = []
    Disc_loss = []
    Gen_acc = []
    Disc_acc = []
    vGen_loss = []
    vDisc_loss = []
    vGen_acc = []
    vDisc_acc = []
    images = []

    for epoch in range (epochs):
        Gloss=0
        Dloss=0
        Gacc=0
        Dacc=0
        count=0   
        print("Epoch",epoch)
        for i, dat in enumerate(train_loader): #trainloader for wehn we load our images  
            (dat_colour, dat_bw) = dat
            fake_label = torch.unsqueeze(torch.from_numpy(np.zeros(len(dat_bw))).float(),1)
            real_label = torch.unsqueeze(torch.from_numpy(np.ones(len(dat_colour))).float(),1)


            optimizerD.zero_grad()

            fake_colour = Generator(dat_bw)

            #discriminator training on real data
            real_pred = Discriminator(dat_colour)
            real_error = loss_fncD(real_pred, real_label) #target not defined yet
            real_error.backward()

            #discriminator train on fake data
            fake_pred = Discriminator(fake_colour) #feeding generator data
            fake_error = loss_fncD(fake_pred, fake_label) #arguments: fake_predictions, fake_labels
            fake_error.backward()

            optimizerD.step()

            Dacc+=(correctCount(real_pred, real_label)+correctCount(fake_pred, fake_label))
            Dloss += float(real_error + fake_error)

            optimizerG.zero_grad()
            fake_colour = Generator(dat_bw)

            #generator Training
            pred_gen = Discriminator(fake_colour)
            gen_error = loss_fncG(pred_gen, real_label)
            gen_error.backward(retain_graph=True)

            optimizerG.step()

            Gacc=correctCount(pred_gen, real_label)
            Gloss+=float(gen_error)
            count=i+1

        Disc_loss.append(Dloss/count) 
        Gen_loss.append(Gloss/count) 
        Disc_acc.append(Dacc/(2*(len(train_loader)*train_loader.batch_size)))
        Gen_acc.append(Gacc/(len(train_loader)*train_loader.batch_size))
        counter.append(epoch+1)

        print("epoch:",epoch+1,"gen loss: ", f'{Gloss/count:.4f}',"disc loss: ", f'{Dloss/count:.4f}',"gen acc: ", f'{Gacc/(len(train_loader)*train_loader.batch_size):.4f}',"disc acc: ", f'{Dacc/(2*(len(train_loader)*train_loader.batch_size)):.4f}')
    
    history = [counter, Gen_loss, Disc_loss,Gen_acc,Disc_acc]
    return history

def plot(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history[0], history[1], label='Generator Loss')
    plt.plot(history[0], history[2], label = 'Discrimnator Loss')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(history[0], history[3], label='Generator Accuracy')
    plt.plot(history[0], history[4], label = 'Discrimnator Accuracy')
    plt.legend()
    
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
    

def main():
    epochs=10
    
    (train_loader, val_loader, test_loader)=dataset(batch_size=64)
    Generator, loss_fncG, optimizerG = initialize_model_G(lr = 0.001)
    Discriminator, loss_fncD, optimizerD = initialize_model_D(lr = 0.001)
    
    history=train(Generator, loss_fncG, optimizerG,Discriminator, loss_fncD, optimizerD, train_loader, val_loader, epochs)
    plot(history)
    
    loss_fnc = nn.MSELoss()
    print("Training Loss:",validate(Generator, train_loader, loss_fnc))
    print("Validation Loss:",validate(Generator, val_loader, loss_fnc))
    print("Test Loss:",validate(Generator, test_loader, loss_fnc))
    
    inference(train_loader, Generator)
    inference(val_loader, Generator)
    
    
if __name__ == "__main__":
    main()