# -*- coding:utf-8 -*-
from torchvision import transforms
from torchvision.datasets import MNIST


def mnist_dataset():
    resize = 24
    path = './data'
    train_dataset = MNIST(path,
                          train=True,
                          transform=transforms.Compose([
                              transforms.Resize(resize),
                              transforms.ToTensor(),
                          ]),
                          download=True)

    val_dataset = MNIST(path,
                        train=False,
                        transform=transforms.Compose([
                            transforms.Resize(resize),
                            transforms.ToTensor()
                        ]),
                        download=True)
    return train_dataset, val_dataset


if __name__ == '__main__':
    pass