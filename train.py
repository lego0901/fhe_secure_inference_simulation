import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from network import *


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="training CIFAR-10")
    parser.add_argument("--data_root", default="./data", help="data path")
    parser.add_argument("--load_path", default=None, help="directory to load model")
    parser.add_argument("--save_path", default=None, help="directory to save model")
    parser.add_argument("--total_epochs", default=20, help="total number of epochs on training", type=int)
    parser.add_argument("--batch_size", default=128, help="batch size for training", type=int)
    parser.add_argument("--block_type", default="naive", help="nonlinearity computation type", type=str, choices=["naive", "chebyshev", "remez"])
    parser.add_argument("--poly_degree", default=10, help="max degree of approx polynomial", type=int)
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_scheduler", default="multistep", help="lr scheduler", type=str, choices=["none", "multistep"])
    parser.add_argument("--activation_decay", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--activation_decay_norm", default=2, type=int, help="Lk norm for computing act decay")
    parser.add_argument("--device", default="cuda", help="device for model")
    parser.add_argument("--use_leaky_relu", default=False, help="to use leaky relu")
    parser.add_argument("--leaky_relu_negative_slope", default=0.1, type=float, help="negative slope")
    parser.add_argument("--clip_before", default=True, help="clip to [-1, 1] before approx")
    parser.add_argument("--manual_seed", default=23, type=int, help="manual seed to avoid randomness")
    # fmt: on

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    net = SimpleMNISTNet(
        n=args.poly_degree,
        block_type=args.block_type,
        use_leaky_relu=args.use_leaky_relu,
        negative_slope=args.leaky_relu_negative_slope,
    ).to(args.device)

    if args.clip_before:
        net.set_clip_before(True)

    if args.load_path:
        if not os.path.exists(args.load_path):
            raise RuntimeError("load path not exists")
        net.load_state_dict(torch.load(args.load_path))

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(
        root=args.data_root, train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    if args.lr_scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.5 * args.total_epochs), int(0.75 * args.total_epochs)],
            gamma=0.1,
        )
    else:
        scheduler = None

    for epoch in range(1, args.total_epochs + 1):
        net.train()

        train_loss, train_correct, train_total = 0.0, 0, 0

        for data, target in dataloader:
            data, target = data.to(args.device), target.to(args.device)
            batch_size = data.shape[0]
            output = net(data)

            loss_orig = criterion(output, target)
            loss_act = net.sum_activation_lk_norm(args.activation_decay_norm)
            loss = loss_orig + loss_act * args.activation_decay

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            _, predicted = output.max(1)
            train_total += batch_size
            train_correct += predicted.eq(target).sum().item()

            print(
                "Epoch {:0>3}/{:0>3}: train loss = {:.4f}, acc = {:.4f}\r".format(
                    epoch,
                    args.total_epochs,
                    train_loss / train_total,
                    train_correct / train_total,
                ),
                end="",
            )

        print()

        if args.save_path:
            torch.save(net.state_dict(), args.save_path)

        if scheduler is not None:
            scheduler.step()
