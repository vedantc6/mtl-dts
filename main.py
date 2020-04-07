import argparse
import torch 
from read_data import Dataset

def main(args):
    conll_path = "./data/datasets/conll04/"
    ade_path = "./data/datasets/ade/"

    device = torch.device('cuda' if args.cuda else 'cpu')

    data = Dataset(conll_path, "conll04", 2, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='use CUDA?')
    args = parser.parse_args()
    main(args)