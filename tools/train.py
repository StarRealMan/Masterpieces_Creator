import os
import sys
sys.path.append("..")
import time
import torch
import torch.nn as nn
import torch.optim as optim
import itertools

import utils.dataset as mydataset
import models.models as mymodel

arg_batchsize = 4
arg_workers = 8
arg_epochs = 50
arg_lr = 2e-4
arg_b1 = 0.5
arg_b2 = 0.999
arg_layer_depth = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(3)

# Define Generator
# monet -> photo
netG_A = mymodel.Generator(3, arg_layer_depth).to(device)
# photo -> monet
netG_B = mymodel.Generator(3, arg_layer_depth).to(device)

# Define Discrimator
netD_A = mymodel.Discriminator(3, arg_layer_depth).to(device)
netD_B = mymodel.Discriminator(3, arg_layer_depth).to(device)

# weight initialization
mymodel.weight_init(netG_A)
mymodel.weight_init(netG_B)
mymodel.weight_init(netD_A)
mymodel.weight_init(netD_B)


# optimizer
netG_optim = optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr = arg_lr, betas = (arg_b1, arg_b2))
netD_optim = optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr = arg_lr, betas = (arg_b1, arg_b2))

GAN_LOSS = nn.MSELoss()
Cycle_LOSS = nn.L1Loss()
Identity_LOSS = nn.L1Loss()

paint_dataset = mydataset.Paint_Dataset("../data/", 256, arg_workers, arg_batchsize)
random_photo_dataset = mydataset.Random_Photo_Dataset("../data/", 256, arg_batchsize)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

train_hist = {}
train_hist['G_losses'] = []
train_hist['D_losses'] = []

print('train is starting')

for epoch in range(arg_epochs):
    t = time.time()
    
    netG_A.train()
    netG_B.train()
    
    netD_A.train()
    netD_B.train()
    
    G_losses = 0
    D_losses = 0
    
    paint_dataloader = paint_dataset.get_dataloader()

    for A in paint_dataloader:
        B = random_photo_dataset.getRandomBatch()
        
        A = A[0].to(device)
        B = B.to(device)
        
        A2B = netG_A(A)
        B2A = netG_B(B)

        A2B2A = netG_B(A2B)
        B2A2B = netG_A(B2A)

        set_requires_grad([netD_A, netD_B], False)

        pred_fake_A = netD_A(B2A)
        pred_fake_B = netD_B(A2B)

        G_GAN_loss = GAN_LOSS(pred_fake_A, torch.ones_like(pred_fake_A)) +\
                        GAN_LOSS(pred_fake_B, torch.ones_like(pred_fake_B))
        G_Cycle_loss = Cycle_LOSS(A2B2A, A) + Cycle_LOSS(B2A2B, B)
        G_identity_loss = Identity_LOSS(netG_A(B), B) + Identity_LOSS(netG_B(A), A)
        G_loss = G_GAN_loss + 10 * G_Cycle_loss #+ 5 * G_identity_loss

        netG_optim.zero_grad()
        G_loss.backward()
        netG_optim.step()
        
        set_requires_grad([netD_A, netD_B], True)

        pred_real_A = netD_A(A)
        pred_fake_A = netD_A(B2A.detach())

        D_A_loss = GAN_LOSS(pred_real_A, torch.ones_like(pred_real_A)) +\
                    GAN_LOSS(pred_fake_A, torch.zeros_like(pred_fake_A))

        pred_real_B = netD_B(B)
        pred_fake_B = netD_B(A2B.detach())

        
        D_B_loss = GAN_LOSS(pred_real_B, torch.ones_like(pred_real_B)) +\
                    GAN_LOSS(pred_fake_B, torch.zeros_like(pred_fake_B))
        
        D_loss = (D_A_loss + D_B_loss) / 2

        netD_optim.zero_grad()
        D_loss.backward()
        netD_optim.step()

        D_losses += D_loss.item() / len(paint_dataloader)
        G_losses += G_loss.item() / len(paint_dataloader)

    print(f'[{epoch + 1}/{arg_epochs}]\tD_loss : {D_losses:.6f}\tG_loss : {G_losses:.6f}\ttime : {time.time() - t:.3f}s')
    
    train_hist['G_losses'].append(G_losses)
    train_hist['D_losses'].append(D_losses)

    # save model per 10epochs
    if (epoch + 1) % 10 == 0:
        if not os.path.exists('../models/model_G'):
            os.makedirs('../models/model_G')

        if not os.path.exists('../models/model_D'):
            os.makedirs('../models/model_D')

        torch.save(netG_A.state_dict, '../models/model_G/' + f'netG(uNet)_A{epoch + 1}.pt')
        torch.save(netG_B.state_dict, '../models/model_G/' + f'netG(uNet)_B{epoch + 1}.pt')

        torch.save(netD_A.state_dict, '../models/model_D/' + f'netD(uNet)_A{epoch + 1}.pt')
        torch.save(netD_B.state_dict, '../models/model_D/' + f'netD(uNet)_B{epoch + 1}.pt')

        print(f'Model is saved at {epoch + 1}epochs')

torch.save(netG_A.state_dict, '../models/model_G/final_model_G_A.pt')
torch.save(netG_B.state_dict, '../models/model_G/final_model_G_B.pt')
torch.save(netD_A.state_dict, '../models/model_D/final_model_D_A.pt')
torch.save(netD_B.state_dict, '../models/model_D/final_model_D_B.pt')