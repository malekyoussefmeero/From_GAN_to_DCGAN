from omegaconf import OmegaConf
import yaml
from models.discriminator import Discriminator
from models.generator import Generator
from train.utils import weights_init,get_noise
import torch 
from tqdm import tqdm
import numpy as np
from data.data import DataloaderDCGAN,get_label
from torchvision.utils import save_image


def training(config ='./configs/config.yaml'):

    with open(config, 'r') as f:
        configuration = OmegaConf.create(yaml.safe_load(f))
    
    #check if GPUs are available(to make training faster)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # define our discriminator and generator 
    # Inialise their weights (as mentionned in the paper)
    gen = Generator(configuration.train.noise_dim).apply(weights_init)
    gen.to(device)
    disc = Discriminator (configuration.train.img_shape).apply(weights_init)
    disc.to(device)


    # define optimizers
    optimizer_gen  = torch.optim.Adam(gen.parameters(),configuration.optimizer.lrG,(0.5,0.999))
    optimizer_disc = torch.optim.Adam(disc.parameters(),configuration.optimizer.lrD,(0.5,0.999))


    # define loss function
    loss = torch.nn.BCELoss()

    # get data for training
    dataset=  DataloaderDCGAN(configuration.data.img_path,configuration.train.img_shape)
    dataloader = torch.utils.data.DataLoader(dataset,configuration.train.batch, shuffle=True)


    # training loop
    for epoch in tqdm(range(configuration.train.epochs)):
        g_loss =[]
        d_loss=[]
        for indx,batch in enumerate(dataloader):

            # make sure all grads are zero
            optimizer_disc.zero_grad()
            optimizer_gen.zero_grad()

            # create labels
            real_label,fake_label = get_label(configuration.train.batch)
            real_label.to(device)
            fake_label.to(device)

                                                ### train the generator network ####

            # create generator_input and generate output
            noise = get_noise(configuration.train.batch,configuration.train.noise_dim).to(device)
            fake_images = gen(noise)

            # classify generator output 
            disc_fake_output = disc(fake_images)

            # Calculate generator_loss
            print(f"here --> {disc_fake_output.size()}")
            generator_loss = loss(disc_fake_output,real_label)
            g_loss.append(generator_loss)

            # Compute  backward pass
            generator_loss.backward()
            optimizer_gen.step()
            
                                                ### train the discriminator network ####

           # Calculate loss for real and fake images
            disc_real_output = disc(batch)
            real_loss = loss(disc_real_output,real_label)
            fake_loss = loss(disc_fake_output,fake_label)
            disc_loss = real_loss + fake_loss
            d_loss.append(disc_loss)

            # Compute  backward pass
            disc_loss.backward()
            optimizer_disc.step()
        
        # print losses
        print(f"epoch : {epoch} generator_loss :{np.mean(g_loss)} discriminator_loss : {np.mean(d_loss)} ")

        # save a the generator output afeter each epoch to visualise the evolution of the training
        save_image(fake_images,"./data/gen_images"+f"img_{epoch}.jpg")
    
    torch.save(gen.state_dict(), './checkpoints/generator.ckpt')


if __name__ =='__main__': 
    training()















            














