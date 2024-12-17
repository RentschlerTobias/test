### AUTOENCODER ### MODEL_C
### MODEL MUST BE IMPLEMENTED IN LEARNING_CENTER CLASSS....
### IMPORT NECESSARY LIBRARIES ###

from torchvision import models
import os
import numpy as np 
import torch
import torch.nn as nn


class auto_encoder_convolution_long(torch.nn.Module):
    def __init__(self, padding_x, padding_y, padding_z):
        super().__init__()
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.padding_z = padding_z
        #in_channels = 4 (Vx, Vy, Vz, p)

        ###ENCODER###
        self.encoder = torch.nn.Sequential(
            nn.Conv3d(in_channels =4, out_channels =3, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =3, out_channels =2, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =2, out_channels =1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1),
            nn.Tanh())


        ###DECODER###
        self.decoder = torch.nn.Sequential(            
            nn.ConvTranspose3d(in_channels =1, out_channels = 1, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[1],self.padding_y[1],self.padding_z[1])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =1, out_channels = 1, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[2],self.padding_y[2],self.padding_z[2])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =1, out_channels =2, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =2, out_channels =3, kernel_size=3, stride=2, padding=1,
                               output_padding=(self.padding_x[3],self.padding_y[3],self.padding_z[3])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =3, out_channels = 4, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[4],self.padding_y[4],self.padding_z[4])),
            nn.Tanh())
                              
    def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def l1_regularization(self):
        l1_loss = 0
        for i in range(len(self.encoder)):
            try:
                l1_loss += torch.sum(torch.abs(self.encoder[i].weight)) + torch.sum(torch.abs(self.decoder[i].weight))
            except:
                pass
        return l1_loss
    
class auto_encoder_convolution_mini(torch.nn.Module):
    def __init__(self, padding_x, padding_y, padding_z):
        super().__init__()
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.padding_z = padding_z
        #in_channels = 4 (Vx, Vy, Vz, p)

        ###ENCODER###
        self.encoder = torch.nn.Sequential(
            nn.Conv3d(in_channels =4, out_channels =3, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =3, out_channels =2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =2, out_channels =1, kernel_size=3, stride=1, padding=1),
            nn.Tanh())

        ###DECODER###
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose3d(in_channels =1, out_channels =2, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =2, out_channels =3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =3, out_channels = 4, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[4],self.padding_y[4],self.padding_z[4])),
            nn.Tanh())
                              
    def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def l1_regularization(self):
        l1_loss = 0
        for i in range(len(self.encoder)):
            try:
                l1_loss += torch.sum(torch.abs(self.encoder[i].weight)) + torch.sum(torch.abs(self.decoder[i].weight))
            except:
                pass
        return l1_loss   


class auto_encoder_convolution_short(torch.nn.Module):
    def __init__(self, padding_x, padding_y, padding_z):
        super().__init__()
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.padding_z = padding_z
        #in_channels = 4 (Vx, Vy, Vz, p)

        ###ENCODER###
        self.encoder = torch.nn.Sequential(
            nn.Conv3d(in_channels =4, out_channels =3, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =3, out_channels =2, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =2, out_channels =1, kernel_size=3, stride=1, padding=1),
            nn.Tanh())


        ###DECODER###
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose3d(in_channels =1, out_channels =2, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =2, out_channels =3, kernel_size=3, stride=2, padding=1,
                               output_padding=(self.padding_x[3],self.padding_y[3],self.padding_z[3])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =3, out_channels = 4, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[4],self.padding_y[4],self.padding_z[4])),
            nn.Tanh())
                              
    def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def l1_regularization(self):
        l1_loss = 0
        for i in range(len(self.encoder)):
            try:
                l1_loss += torch.sum(torch.abs(self.encoder[i].weight)) + torch.sum(torch.abs(self.decoder[i].weight))
            except:
                pass
        return l1_loss        

class auto_encoder_convolution_medium(torch.nn.Module):
    def __init__(self, padding_x, padding_y, padding_z):
        super().__init__()
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.padding_z = padding_z
        #in_channels = 4 (Vx, Vy, Vz, p)

        ###ENCODER###
        self.encoder = torch.nn.Sequential(
            nn.Conv3d(in_channels =4, out_channels =3, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =3, out_channels =2, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =2, out_channels =1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1),
            nn.Tanh())


        ###DECODER###
        self.decoder = torch.nn.Sequential(            
            nn.ConvTranspose3d(in_channels =1, out_channels = 1, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[2],self.padding_y[2],self.padding_z[2])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =1, out_channels =2, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =2, out_channels =3, kernel_size=3, stride=2, padding=1,
                               output_padding=(self.padding_x[3],self.padding_y[3],self.padding_z[3])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =3, out_channels = 4, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[4],self.padding_y[4],self.padding_z[4])),
            nn.Tanh())
                              
    def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def l1_regularization(self):
        l1_loss = 0
        for i in range(len(self.encoder)):
            try:
                l1_loss += torch.sum(torch.abs(self.encoder[i].weight)) + torch.sum(torch.abs(self.decoder[i].weight))
            except:
                pass
        return l1_loss

class auto_encoder_convolution_scae(torch.nn.Module):
    def __init__(self, padding_x, padding_y, padding_z):
        super().__init__()
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.padding_z = padding_z

        ###ENCODER###
        self.encoder = torch.nn.Sequential(
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1),
            nn.Tanh())


        ###DECODER###
        self.decoder = torch.nn.Sequential(            
            nn.ConvTranspose3d(in_channels =1, out_channels = 1, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[1],self.padding_y[1],self.padding_z[1])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =1, out_channels = 1, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[2],self.padding_y[2],self.padding_z[2])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =1, out_channels =1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =1, out_channels =1, kernel_size=3, stride=2, padding=1,
                               output_padding=(self.padding_x[3],self.padding_y[3],self.padding_z[3])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =1, out_channels = 1, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[4],self.padding_y[4],self.padding_z[4])),
            nn.Tanh())
                              
    def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def l1_regularization(self):
        l1_loss = 0
        for i in range(len(self.encoder)):
            try:
                l1_loss += torch.sum(torch.abs(self.encoder[i].weight)) + torch.sum(torch.abs(self.decoder[i].weight))
            except:
                pass
        return l1_loss


class auto_encoder_convolution(torch.nn.Module):
    def __init__(self, padding_x, padding_y, padding_z):
        super().__init__()
        self.padding_x = padding_x
        self.padding_y = padding_y
        self.padding_z = padding_z
        #in_channels = 4 (Vx, Vy, Vz, p)

        ###ENCODER###
        self.encoder = torch.nn.Sequential(
            nn.Conv3d(in_channels =4, out_channels =3, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =3, out_channels =2, kernel_size=3, stride=2, padding=1),
            nn.Tanh())


        ###DECODER###
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose3d(in_channels =2, out_channels =3, kernel_size=3, stride=2, padding=1,
                               output_padding=(self.padding_x[1],self.padding_y[1],self.padding_z[1])),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels =3, out_channels = 4, kernel_size=3, stride=2, padding=1, 
                               output_padding = (self.padding_x[2],self.padding_y[2],self.padding_z[2])),
            nn.Tanh())
                              
    def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def l1_regularization(self):
        l1_loss = 0
        for i in range(len(self.encoder)):
            try:
                l1_loss += torch.sum(torch.abs(self.encoder[i].weight)) + torch.sum(torch.abs(self.decoder[i].weight))
            except:
                pass
        return l1_loss

class SCAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1

        ###ENCODER###
        self.encoder = torch.nn.Sequential(
            nn.ConvTranspose3d(in_channels = 4, out_channels = 5, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels = 5, out_channels = 6, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels = 6, out_channels = 7, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels = 7, out_channels = 8, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh())


        ###DECODER###
        self.decoder = torch.nn.Sequential(
            nn.Conv3d(in_channels =8, out_channels =7, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =7, out_channels =6, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =6, out_channels =5, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv3d(in_channels =5, out_channels =4, kernel_size=3, stride=1, padding=1),
            nn.linear())
                              
    def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


    def l1_regularization(self):
        l1_loss = 0
        for i in range(len(self.encoder)):
            try:
                l1_loss += torch.sum(torch.abs(self.encoder[i].weight)) + torch.sum(torch.abs(self.decoder[i].weight))
            except:
                pass
        return l1_loss



class neural_network_supervised(nn.Module):
    def __init__(self, cluster_numbers):
        super().__init__()
        self.objective_size = 30
        self.hidden_size_1 = 34
        self.hidden_size_2 = 30
        self.hidden_size_3 = 35
        self.hidden_size_4 = 40
        self.hidden_size_5 = 40
        self.anzahl_cluster = cluster_numbers


        self.layer1 = nn.Linear(self.objective_size, self.hidden_size_1)
        self.layer2 = nn.Linear(self.hidden_size_1, self.anzahl_cluster)
        self.layer3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.layer4 = nn.Linear(self.hidden_size_3, self.hidden_size_4)
        self.layer5 = nn.Linear(self.hidden_size_4, self.hidden_size_5)
        self.layer6 = nn.Linear(self.hidden_size_5, self.anzahl_cluster)
        self.layer7 = nn.Softmax(dim = 1)
        self.acti = nn.ReLU()

    def forward(self, x):
        #x = torch.sigmoid(self.layer1(x))
        #x = torch.sigmoid(self.layer2(x))
        #x = torch.sigmoid(self.layer3(x))
        #x = torch.sigmoid(self.layer4(x))
        #x = torch.sigmoid(self.layer5(x))
        x = self.acti(self.layer1(x))
        x = self.acti(self.layer2(x))
        x = self.layer7(x)
        return x





