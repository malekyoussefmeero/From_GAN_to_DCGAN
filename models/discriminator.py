"Here we will define our discriminator network"

from math import tanh
import torch.nn as nn
import torch
class Discriminator(nn.Module):
    def __init__(self,input_dim) -> None:
        super().__init__()
        self.network = nn.Sequential(

            # First bloc
            nn.Conv2d(input_dim,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            #  Second bloc
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            # Third bloc
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            # Fourth bloc
            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            # Last bloc
            nn.Conv2d(512,1,4,1,0),
            nn.Tanh()
        )

    def forward(self, x):
            return self.network(x)

"Quick visualisation of our generator "

if __name__ =='__main__': 
	device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
	disc = Discriminator(3).to(device)
	print(disc)
