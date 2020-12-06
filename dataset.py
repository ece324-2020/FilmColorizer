import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

#CombinedDataset of Coloured and B&W stills from Casablanca
class CombinedDataset(data.Dataset):
    def __init__(self, Color, BW):
        self.Color=Color
        self.BW=BW
        
        #Transform to change image into tensor, then adjust pixel values to between [-1,1]
        self.transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.Color)

    def __getitem__(self, index):
        Color=self.Color[index]
        BW=self.BW[index]
        
        Color=self.transform(Color)
        BW=self.transform(BW)
        return Color, BW
    
#New version of dataset that splits the train, validation and test set sequentially
def dataset(batch_size=64):
    #Specify directories for coloured and BW datasets
    Colored_root="./Colored"
    BW_root="./BW"

    #Retrieve Colored Images
    Colored_img_list = list(Path(Colored_root).rglob("*.[jJ][pP][eE][gG]"))
    Colored_imgs = []
    for img in Colored_img_list:
        Colored_imgs+=[plt.imread(img)]
    
    #Retrieve Black and White Images
    BW_img_list = list(Path(BW_root).rglob("*.[jJ][pP][eE][gG]"))
    BW_imgs = []
    for img in BW_img_list:
        BW_imgs+=[plt.imread(img)]
    
    #Specify Sequential Range for each dataset
    Colored_imgs_train=Colored_imgs[0:3999]
    BW_imgs_train=BW_imgs[0:3999]
    Colored_imgs_val=Colored_imgs[4000:4999]
    BW_imgs_val=BW_imgs[4000:4999]
    Colored_imgs_test=Colored_imgs[5000:-1]
    BW_imgs_test=BW_imgs[5000:-1]
    
    #Create dataset and dataloader objects
    train_dataset=CombinedDataset(Colored_imgs_train, BW_imgs_train)
    val_dataset=CombinedDataset(Colored_imgs_val, BW_imgs_val)
    test_dataset=CombinedDataset(Colored_imgs_test, BW_imgs_test)

    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader=DataLoader(val_dataset, batch_size=batch_size)
    test_loader=DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


#Old version of dataset that used train_test_split to randomly split the dataset into train and validation datasets
def dataset_old(batch_size=4, seed=0):
    #Specify directories for coloured and BW datasets
    Colored_root="./Colored"
    BW_root="./BW"

    Colored_img_list = list(Path(Colored_root).rglob("*.[jJ][pP][gG]"))
    Colored_imgs = []
    for img in Colored_img_list:
        Colored_imgs+=[plt.imread(img)]
    
    BW_img_list = list(Path(BW_root).rglob("*.[jJ][pP][gG]"))
    BW_imgs = []
    for img in BW_img_list:
        BW_imgs+=[plt.imread(img)]

    Colored_imgs_train, Colored_imgs_test, BW_imgs_train, BW_imgs_test = train_test_split(Colored_imgs, BW_imgs, test_size=0.2, random_state=seed)
    
    train_dataset=CombinedDataset(Colored_imgs_train, BW_imgs_train)
    test_dataset=CombinedDataset(Colored_imgs_test, BW_imgs_test)
    
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader=DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader
