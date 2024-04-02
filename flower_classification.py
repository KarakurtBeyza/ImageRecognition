
input="flowers"
output="data"
splitfolders.ratio(input,output,seed=42,ratio=(0.6,0.2,0.2))

"""##2.Creating Datapaths & setting the device to work easier"""

import os
from pathlib import Path
data_path=Path("data/")
train_dir=data_path /"train"
test_dir=data_path/"test"
val_dir=data_path/"val"
device="cuda" if torch.cuda.is_available() else "cpu"

"""##3.Transforming the data  """

import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data_transform=transforms.Compose([#Resize the image to 224x224 so it suits the resnet50 input size
                                   transforms.Resize(size=(224,224)),
                                   #Flip the images randomly on the horizontal
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   #Transform the image to tensor so it is actually useful for the training model
                                   transforms.ToTensor()
])
# nn.sequential is also optional to the compose method,it  does the same thing

"""##4.Loading the data using ImageFolder-->Torchvision base class for custom dataset"""

from torchvision import datasets
train_data=datasets.ImageFolder(root=train_dir,
                                transform=data_transform,#transform for the data
                                target_transform=None) #transform for the target/label
test_data=datasets.ImageFolder(root=test_dir,
                               transform=data_transform,
                               target_transform=None)
#Checking the data size
print(len(train_data.samples))
print(len(test_data.samples))

"""##5.Creating the dataloaders"""

train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=10,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_data,
                                       batch_size=10,
                                       shuffle=False
                                       )

"""##6.Creating the resnet50 object with pretrained model in the models lib"""

import torchvision.models as models
resnet50=models.resnet50(pretrained=True)

#Changing the output features in the last layer of resnet50 model
resnet50.fc = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=2048,
        out_features=5
   )
)

"""##7.Creating the train loop for the model training step"""

def train(model,dataloader,loss_fn,optimizer,device=device):
  model.train()
  train_loss=0.0#
  train_accuracy=0.0#
  iter_loss=0.0#
  iter_acc=0.0#
  iterations=0#

  for i,(inputs,labels) in enumerate(dataloader):
    #Sending the inputs and the model to the same device
    inputs=inputs.to(device)
    labels=labels.to(device)
    model=model.to(device)

    outputs=model(inputs)

    #Loss function
    loss=loss_fn(outputs,labels)
    iter_loss +=loss.item()

    #Optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _,predicted=torch.max(outputs,1)
    iter_acc+=(predicted==labels).sum()
    iterations+=1

  train_loss=iter_loss/iterations
  train_accuracy=(100*iter_acc/len(train_data))


  return train_loss,train_accuracy

"""##8.Creating the test loop for the testing step"""

def test(model,dataloader,loss_fn,optimizer,device=device):
  model.eval()
  test_loss=0.0
  test_accuracy=0.0
  iter_loss=0.0
  iter_acc=0.0
  iterations=0

  for i ,(inputs,labels) in enumerate(dataloader):
    inputs=inputs.to(device)
    labels=labels.to(device)
    model=model.to(device)

    outputs=model(inputs)
    loss=loss_fn(outputs,labels)
    iter_loss=loss.item()

    _,predicted=torch.max(outputs,1)
    iter_acc+=(predicted==labels).sum()
    iterations+=1
  test_loss=(iter_loss/iterations)
  test_accuracy=(100*iter_acc/len(test_data))
  return test_loss,test_accuracy

"""##9.Creating the general function to train and test the resnet50 model"""

def process(model,train_dataloader,test_dataloader,optimizer,loss_fn,
            epoches,device=device):

  results={"train_loss":[],
           "train_accuracy":[],
           "test_loss":[],
           "test_accuracy":[]
           }
  for epoch in range(epoches):
    train_loss,train_accuracy=train(model,
                                    train_dataloader,
                                    loss_fn,
                                    optimizer)
    test_loss,test_accuracy=test(model,
                                    test_dataloader,
                                    loss_fn,
                                    optimizer)


    print('Epoch {}/{}, Training loss:{:.3f},Training Accuracy: {: .3f},Testing Loss:{: .3f},Testing Accuracy:{: .3f}'
  .format(epoch+1,epoches,train_loss,train_accuracy,test_loss,test_accuracy))
    results["train_loss"].append(train_loss)
    results["train_accuracy"].append(train_accuracy)
    results["test_loss"].append(test_loss)
    results["test_accuracy"].append(train_loss)

"""##10.Calling the main function and training the model"""

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(resnet50.parameters(),lr=0.001)
process(resnet50,train_loader,test_loader,optimizer,loss_fn,10)