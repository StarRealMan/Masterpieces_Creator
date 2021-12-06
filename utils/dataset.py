import os
import random
import numpy as np

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Paint_Dataset():
    def __init__(self, dataroot, image_size, workers, batchsize):
        self.style_dataset = dset.ImageFolder(root = os.path.join(dataroot, "monet_jpg"),
                                    transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

        self.style_datasetFlip = dset.ImageFolder(root = os.path.join(dataroot, "monet_jpg"),
                                    transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.RandomHorizontalFlip(1)
                                ]))

        assert self.style_dataset
        assert self.style_datasetFlip

        self.style_dataloader = torch.utils.data.DataLoader(self.style_dataset + self.style_datasetFlip,
                                                            batch_size=batchsize, shuffle=True, num_workers=int(workers))

    def get_dataloader(self):
        return self.style_dataloader

class Random_Photo_Dataset():
    def __init__(self, dataroot, image_size, batchsize, manualSeed = None):
        self.batchsize = batchsize
        self.random_dataset = dset.ImageFolder(root = os.path.join(dataroot, "photo_jpg"),
                                    transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        
        assert self.random_dataset

        if manualSeed is None:
            manualSeed = random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        np.random.seed(manualSeed)

    def getRandomBatch(self):
        rand_num = np.random.randint(0, len(self.random_dataset) - 1, self.batchsize)
        batch_data = torch.stack([self.random_dataset[i][0] for i in rand_num], 0)

        return batch_data


if __name__ == "__main__":
    img_size = 256
    myDataset = Random_Photo_Dataset("/home/starydy/Develop/Masterpieces_Creator/data/", img_size, 8)
    data = myDataset.getRandomBatch()
    print(data.shape)
    vutils.save_image(data, 'dataset.png', normalize=True)
