import argparse
import torch
from read_data import Dataset
from model import MTLArchitecture
from utils import get_init_weights
import copy
import random
import math
from logger import Logger

def main(args):
    if args.dataset_name == "conll04":
        path = "./data/datasets/conll04/"
    else:
        path = "./data/datasets/ade/"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger = Logger(args.model + '.log', True)

    device = torch.device('cuda' if args.cuda else 'cpu')

    data = Dataset(path, args.dataset_name, 2, device)

    data.log(logger)
    logger.log(str(args))

    model = MTLArchitecture(len(data.word2x), args.shared_layer_size, len(data.char2c), args.char_dim, \
                            args.hidden_dim, args.dropout, args.re_dropout, args.num_layers_shared, \
                            args.num_layers_ner, args.num_layers_re, len(data.tag2y), \
                            len(data.relation2y), args.init, args.label_embeddings_size, \
                            args.re_f1_size, args.re_lambda, args.e1_activation_type, \
                            args.r1_activation_type, args.recurrent_unit, device).to(device)

    model.apply(get_init_weights(args.init))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_model = copy.deepcopy(model)
    best_perf = float('-inf')
    bad_epochs = 0

    try:
        for ep in range(1, args.epochs + 1):
            random.shuffle(data.batches_train)
            output = model.do_epoch(ep, data.batches_train, args.clip, optim, logger=logger, check_interval=args.check_interval)

            if math.isnan(output['loss']):
                break

            with torch.no_grad():
                    eval_result = model.evaluate(data.batches_test, logger=logger, tag2y=data.tag2y, rel2y=data.relation2y)
                    print(eval_result)
                
            # perf = eval_result['f1_<all>'] + eval_result['re_f1']

            logger.log('Epoch {:3d} | '.format(ep) + ' '.join(['{:s} {:8.3f} | '.format(key, output[key])
                                 for key in output]), newline=False)
            torch.save({'opt': args, 'sd': model.state_dict()}, args.model+'models/model_epoch_' + str(ep))
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
        logger.log('-' * 89)
        logger.log('Exiting from training early')

    return best_model, best_perf

# def meta_main(args):
#     if not args.train:
#         val_result, test_result = run_tests(args)
#         print('Validation')
#         for key in val_result:
#             print('{:20s}: {:8.3f}'.format(key, val_result[key]))
#         print()
#         print('Test')
#         for key in test_result:
#             print('{:20s}: {:8.3f}'.format(key, test_result[key]))
#         exit()

#     logger = Logger(args.model + '.log', True)
#     perfs = []
#     best_perf = float('-inf')
#     best_args = None

#     hypers = OrderedDict({
#         'batch_size': [1, 2, 4, 8, 16, 32],
#         'wdim': [50, 100, 150, 200],
#         'cdim': [10, 25, 50],
#         'hdim': [50, 100, 150, 200],
#         'lr': [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
#         'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#         'clip': [1, 5, 10],
#         'init': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
#         'seed': list(range(100000)),
#     })

#     for run_number in range(1, args.nruns + 1):

#         # If random runs, scramble hyperparameters.
#         if args.nruns > 1:
#             logger.log('RUN NUMBER: %d' % (run_number))
#             for hyp, choices in hypers.items():
#                 choice = choices[torch.randint(len(choices), (1,)).item()]
#                 assert hasattr(args, hyp)
#                 args.__dict__[hyp] = choice

#         model, perf = main(args, logger)

#         perfs.append(perf)
#         if perf > best_perf:
#             best_perf = perf
#             best_args = copy.deepcopy(args)
#             logger.log('------New best validation performance: %g' % perf)
#             logger.log('Saving model to %s' % args.model)
#             torch.save({'opt': args, 'sd': model.state_dict(), 'perf': perf},
#                        args.model)

#     logger.log_perfs(perfs, best_args)

#     val_result, test_result = run_tests(args)

#     logger.log('Validation')
#     for key in val_result:
#         logger.log('{:20s}: {:8.3f}'.format(key, val_result[key]))
#     logger.log('')
#     logger.log('Test')
#     for key in test_result:
#         logger.log('{:20s}: {:8.3f}'.format(key, test_result[key]))


# def run_tests(args):
#     device = torch.device('cuda' if args.cuda else 'cpu')
#     dat = TaggingDataset(args.data, 8, device)

#     package = torch.load(args.model) if args.cuda else \
#               torch.load(args.model, map_location=torch.device('cpu'))
#     opt = package['opt']
#     model = BiLSTMTagger(len(dat.word2x), len(dat.tag2y), len(dat.char2c),
#                          opt.wdim, opt.cdim, opt.hdim, opt.dropout,
#                          opt.layers, opt.nochar, opt.loss, opt.init).to(device)
#     model.load_state_dict(package['sd'])

#     with torch.no_grad():
#         val_result = model.evaluate(dat.batches_val, dat.tag2y)
#         test_result = model.evaluate(dat.batches_test, dat.tag2y)

#     return val_result, test_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./', help='model path')
    parser.add_argument('--cuda', action='store_true', help='use CUDA?')
    parser.add_argument('--dataset_name', default="conll04")
    parser.add_argument('--shared_layer_size', type=int, default=256)
    parser.add_argument('--char_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--re_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers_shared', type=int, default=1)
    parser.add_argument('--num_layers_ner', type=int, default=1)
    parser.add_argument('--num_layers_re', type=int, default=2)
    parser.add_argument('--e1_activation_type', default='tanh', help='activation for NER FF1 [%(default)g]')
    parser.add_argument('--r1_activation_type', default='relu', help='activation for RE FF1 [%(default)g]')
    parser.add_argument('--recurrent_unit', default='gru')
    parser.add_argument('--init', type=float, default=0.01, help='uniform init range [%(default)g]')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate [%(default)g]')
    parser.add_argument('--epochs', type=int, default=10, help='max number of epochs [%(default)d]')
    parser.add_argument('--check_interval', type=int, default=10, metavar='CH',
                        help='number of updates for a check [%(default)d]')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping [%(default)g]')
    parser.add_argument('--label_embeddings_size', type=int, default=25, help='label embedding size [%(default)g]')
    parser.add_argument('--re_lambda', type=int, default=5, help='RE loss parameter')
    parser.add_argument('--re_f1_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42, help='random seed [%(default)d]')
    args = parser.parse_args()
    main(args)