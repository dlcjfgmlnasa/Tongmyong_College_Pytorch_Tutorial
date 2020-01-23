# -*- coding:utf-8 -*-
import os
import torch
import argparse
import torch.optim as opt
from torch.utils.data import DataLoader
from mnist.data_helper import mnist_dataset
from mnist.models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--print_step', type=int, default=10)
    parser.add_argument('--validation_step', type=int, default=200)
    parser.add_argument('--store_step', type=int, default=100)
    parser.add_argument('--model', type=str, choices=['DNN', 'CNN', 'VGG_D'], default='VGG_D')
    parser.add_argument('--store_path', type=str, default=os.path.join('.', 'store'))
    return parser.parse_args()


def main(arguments):
    train_dataset, validation_dataset = mnist_dataset()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=arguments.batch_size,
                              shuffle=True)
    val_dataset_size = len(validation_dataset)
    validation_loader = DataLoader(dataset=validation_dataset,
                                   batch_size=int(val_dataset_size / 100),
                                   shuffle=False)

    if arguments.model == 'DNN':
        model = DNNMnistClassify()
    elif arguments.model == 'CNN':
        model = CNNMnistClassify()
    elif arguments.model == 'VGG_D':
        model = VGG(vgg_type='VGG_D')
    else:
        raise NotImplemented()

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(
        params=model.parameters(),
        lr=arguments.learning_rate
    )

    n_iter = 0
    for epoch in range(arguments.epochs):
        for i, data in enumerate(train_loader):
            train_image, train_label = data
            output = model(train_image)

            loss = criterion(output, train_label)
            accuracy = torch.sum(
                torch.eq(
                    output.argmax(dim=-1),
                    train_label
                ), dtype=torch.float64) / arguments.batch_size
            n_iter += 1

            if i % arguments.validation_step == 0:
                with torch.no_grad():
                    total_val_loss, total_val_accuracy = 0, 0
                    count = 0
                    for val_data in validation_loader:
                        val_image, val_label = val_data
                        val_output = model(val_image)
                        val_loss = criterion(val_output, val_label)
                        total_val_loss += val_loss
                        total_val_accuracy += torch.sum(
                            torch.eq(
                                val_output.argmax(dim=-1),
                                val_label
                            ), dtype=torch.float64) / int(val_dataset_size / 100)
                        count += 1
                    total_val_loss /= count
                    total_val_accuracy /= count

                print("[ Val ] => "
                      "[Epoch] : {0}\t"
                      "[Iter]: {1}\t"
                      "[train_loss]: {2:.2f}\t[train_accuracy]: {3:.2f}\t"
                      "[val_loss]: {4:.2f}\t[val_accuracy]: {5:.2f}".format(
                        epoch, i, loss.item(), accuracy.item(),
                        total_val_loss.item(),total_val_accuracy.item()
                    ))

            if i % arguments.print_step == 0:
                print("[Train] => "
                      "[Epoch] : {0}\t"
                      "[Iter]: {1}\t"
                      "[train_loss]: {2:.2f}\t[train_accuracy]: {3:.2f}\t".
                      format(epoch, i, loss.item(), accuracy.item()))

            if i % arguments.store_step == 0:
                filename = os.path.join(arguments.store_path, '{0}_{1: 05d}.pth'.format(arguments.model, n_iter))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'accuracy': accuracy.item(),
                    'model': arguments.model,
                }, filename)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    args = get_args()
    main(args)