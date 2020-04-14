import argparse
import torch
from read_data import Dataset
from model import MTLArchitecture
from utils import get_init_weights
import copy
import random
import math


def main(args):
    if args.dataset_name == "conll04":
        path = "./data/datasets/conll04/"
    else:
        path = "./data/datasets/ade/"

    device = torch.device('cuda' if args.cuda else 'cpu')

    data = Dataset(path, args.dataset_name, 2, device)

    model = MTLArchitecture(len(data.word2x), args.shared_layer_size, len(data.char2c), args.char_dim, \
                            args.hidden_dim, args.dropout, args.num_layers_shared, args.num_layers_ner, \
                            args.num_layers_re, len(data.tag2y), len(data.relation2y), args.init, \
                            args.label_embeddings_size, args.activation_type, args.recurrent_unit, \
                            device).to(device)

    model.apply(get_init_weights(args.init))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_model = copy.deepcopy(model)
    best_perf = float('-inf')
    bad_epochs = 0

    try:
        for ep in range(1, args.epochs + 1):
            random.shuffle(data.batches_train)
            output = model.do_epoch(ep, data.batches_train, args.clip, optim, check_interval=args.check_interval)

            if math.isnan(output['loss']):
                break

            # with torch.no_grad():
            #     eval_result = model.evaluate(data.batches_val, data.tag2y)

            # perf = eval_result['acc'] if not 'O' in data.tag2y else \
            #        eval_result['f1_<all>']

            # logger.log('Epoch {:3d} | '.format(ep) +
            #            ' '.join(['{:s} {:8.3f} | '.format(key, output[key])
            #                      for key in output]) +
            #            ' val perf {:8.3f}'.format(perf), newline=False)

            # if perf > best_perf:
            #     best_perf = perf
            #     bad_epochs = 0
            #     logger.log('\t*Updating best model*')
            #     best_model.load_state_dict(model.state_dict())
            # else:
            #     bad_epochs += 1
            #     logger.log('\tBad epoch %d' % bad_epochs)

            # if bad_epochs >= args.max_bad_epochs:
            #     break

    except KeyboardInterrupt:
        print('Exiting from training early')

    return best_model, best_perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='use CUDA?')
    parser.add_argument('--dataset_name', default="conll04")
    parser.add_argument('--shared_layer_size', type=int, default=128)
    parser.add_argument('--char_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers_shared', type=int, default=1)
    parser.add_argument('--num_layers_ner', type=int, default=1)
    parser.add_argument('--num_layers_re', type=int, default=1)
    parser.add_argument('--activation_type', default='relu')
    parser.add_argument('--recurrent_unit', default='gru')
    parser.add_argument('--init', type=float, default=0.01, help='uniform init range [%(default)g]')
    parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate [%(default)g]')
    parser.add_argument('--epochs', type=int, default=10, help='max number of epochs [%(default)d]')
    parser.add_argument('--check_interval', type=int, default=10, metavar='CH',
                        help='number of updates for a check [%(default)d]')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping [%(default)g]')
    parser.add_argument('--label_embeddings_size', type=float, default=25, help='label embedding size [%(default)g]')
    args = parser.parse_args()
    main(args)