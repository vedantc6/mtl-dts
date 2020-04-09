import argparse
import torch 
from read_data import Dataset

def main(args):
    if args.dataset_name == "conll04":
        path = "./data/datasets/conll04/"
    else:
        path = "./data/datasets/ade/"

    device = torch.device('cuda' if args.cuda else 'cpu')

    data = Dataset(path, args.dataset_name, 2, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='use CUDA?')
    parser.add_argument('--dataset_name', default="conll04")
    args = parser.parse_args()
    main(args)