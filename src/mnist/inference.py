# -*- coding:utf-8 -*-
import os
import argparse
from mnist.models import *
from mnist.data_helper import mnist_dataset
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store_path', type=str, default=os.path.join('.', 'store'))
    parser.add_argument('--store_name', type=str, default=' 0101.pth')
    return parser.parse_args()


def main(arguments):
    model_path = os.path.join(arguments.store_path, arguments.store_name)
    model_info = torch.load(model_path)

    model_type = model_info['model']
    if model_type == 'DNN':
        model = DNNMnistClassify()
    elif model_type == 'CNN':
        model = CNNMnistClassify()

    model.load_state_dict(model_info['model_state_dict'])
    model.eval()

    _, val_dataset = mnist_dataset()
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            shuffle=False)
    for data in val_loader:
        image, label = data
        output = model(image)
        output = output.argmax(dim=-1)
        result = torch.eq(output, label)
        print('Predicate: {} \t Real: {} \t Result: {}'.format(output.item(), label.item(), result.item()))


if __name__ == '__main__':
    args = get_args()
    main(args)
