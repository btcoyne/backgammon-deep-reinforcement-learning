from Data_Loaders import Data_Loaders
from Networks1 import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(no_epochs):

    batch_size = 128
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    model.to(device)
    model.load_state_dict(torch.load())
    loss_function = nn.MSELoss()
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    losses = []
    loss_amount = []
    #min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    #losses.append(min_loss)
    previous_state = torch.zeros([batch_size,6])
    #data = torch.zeros()
    tensor_input_list = []
    tensor_label_list = []
    for idx, sample in enumerate(data_loaders.train_loader):
        if batch_size == sample['input'].shape[0]:
            tensor_input_list.append(sample['input'])
            tensor_label_list.append(sample['label'])
            #print(type(sample['input']))
            #dataloader[idx]['input'] = sample['label'].to(device)
    input_tensors = torch.stack(tensor_input_list)
    input_tensors = input_tensors.to(device)
    label_tensors = torch.stack(tensor_label_list)
    label_tensors = label_tensors.to(device)

   

    for epoch_i in range(no_epochs):
        running_loss = 0.0
        model.train()
        for in_tensor, lab_tensor in zip(input_tensors, label_tensors):
            optimizer.zero_grad()
            #input = torch.cat((previous_state,sample['input']),1)
            output = model(in_tensor) 

            loss = loss_function(output, lab_tensor)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / (input_tensors.shape[1])
        print("Epoch" + str(epoch_i) + "    " + str(epoch_loss)  )