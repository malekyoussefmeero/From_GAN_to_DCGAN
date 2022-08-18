"Here we will define our generator network"

import torch.nn as nn
import torch
class Generator(nn.Module):
	def __init__(self, noise_dim,output_shape = 64,output_dim=3):
		super(Generator, self).__init__()
		self.noise_dim = noise_dim
		self.output_shape = output_shape
		self.output_dim = output_dim
		self.network = nn.Sequential(
			# first bloc
			nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0, bias=False),    # noise input and values taken from the official paper
            nn.BatchNorm2d(1024),
            nn.ReLU(True),


			## second bloc
			nn.ConvTranspose2d(1024, 512, 8, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

			## third bloc
			nn.ConvTranspose2d(512,256, 5, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

			## fourth bloc
			nn.ConvTranspose2d(256,128, 38, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

			## Fifth bloc
			nn.ConvTranspose2d(128, self.output_dim ,5,2, 1, bias=False),
			nn.Tanh()
		)


	def forward(self, x):
         return self.network(x)


"Quick visualisation of our generator "

if __name__ =='__main__': 
	device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
	gen = Generator(100).to(device)
	print(gen)


