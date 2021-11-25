import torch

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import json

from network import *


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="training CIFAR-10")
    parser.add_argument("--data_root", default="./data", help="data path")
    parser.add_argument("--load_path", help="directory to load model", required=True)
    parser.add_argument("--batch_size", default=128, help="batch size for training", type=int)
    parser.add_argument("--block_type", default="naive", help="nonlinearity computation type", type=str, choices=["naive", "chebyshev", "remez"])
    parser.add_argument("--poly_degree", default=10, help="max degree of approx polynomial", type=int)
    parser.add_argument("--device", default="cuda", help="device for model")
    parser.add_argument("--use_leaky_relu", default=False, help="to use leaky relu")
    parser.add_argument("--leaky_relu_negative_slope", default=0.1, type=float, help="negative slope")
    parser.add_argument("--json_path", default=None, help="json file to store output")
    # fmt: on

    args = parser.parse_args()
    print(args)

    net = SimpleMNISTNet(
        n=args.poly_degree,
        block_type=args.block_type,
        use_leaky_relu=args.use_leaky_relu,
        negative_slope=args.leaky_relu_negative_slope,
    ).to(args.device)

    net.set_use_profiling_layers(True)

    if not os.path.exists(args.load_path):
        raise RuntimeError("load path not exists")
    net.load_state_dict(torch.load(args.load_path))

    transform = transforms.ToTensor()
    testset = torchvision.datasets.MNIST(
        root=args.data_root, train=False, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    net.eval()
    test_correct, test_total = 0, 0

    for data, target in dataloader:
        data, target = data.to(args.device), target.to(args.device)
        batch_size = data.shape[0]
        output = net(data)

        test_total += batch_size
        _, predicted = output.max(1)
        test_correct += predicted.eq(target).sum().item()

    print("Test acc = {:.4f}".format(test_correct / test_total))

    print("Profiling activation magnitudes:")
    profiling_results = net.get_profiling_results()
    total_results = 0
    for name in profiling_results.keys():
        print(name, list(profiling_results[name]))
        total_results += profiling_results[name]
        profiling_results[name] = list(map(int, profiling_results[name]))

    total_results = list(map(int, total_results))
    print("total", total_results)

    if args.json_path is not None:
        eval_results = {
            "acc": test_correct / test_total,
            "layerwise": profiling_results,
            "total": total_results,
        }
        with open(args.json_path, "w") as json_file:
            json.dump(eval_results, json_file)
            json_file.close()
