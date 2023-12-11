import cfgs.config as config
import argparse, yaml
import random
from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser(description='Bilinear Args')

    parser.add_argument('--run', dest='run_mode',
                        choices=['train', 'val', 'test', 'fewshot', 'zeroshot', 'twostage'],
                        help='{train, val, test}',
                        type=str, required=True)

    parser.add_argument('--model', dest='model',
                        help='{bert, ...}',
                        default='bert', type=str)

    parser.add_argument('--dataset', dest='dataset',
                        choices=['imdb2a', 'imdb2b',
                                 'yelp2a', 'yelp2b',
                                 'gdrda', 'gdrdb',
                                 'ppra', 'pprb'],
                        default='imdb2a', type=str)

    parser.add_argument('--gpu', dest='gpu',
                        help="gpu select, eg.'0, 1, 2'",
                        type=str,
                        default="0, 1")

    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        )

    parser.add_argument('--nop',
                        action='store_true',
                        default=False,
                        )

    parser.add_argument('--seed', dest='seed',
                        help='fix random seed',
                        type=int,
                        default=random.randint(0, 99999999))

    parser.add_argument('--version', dest='version',
                        help='version control',
                        type=str,
                        default="default")

    parser.add_argument('--ft',
                        action='store_true',
                        default=False,
                        )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    __C = config.__C

    args = parse_args()
    try:
        cfg_file = "cfgs/{}_model.yml".format(args.model)
        with open(cfg_file, 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        args_dict = edict({**yaml_dict, **vars(args)})
    except:
        args_dict = edict({**vars(args)})
    config.add_edit(args_dict, __C)
    config.proc(__C)

    print('Hyper Parameters:')
    config.config_print(__C)

    from trainer.trainer import PLMTrainer
    if not __C.ft:
        print("Parameter-efficient Fine Tuning...")
        trainer = PLMTrainer(__C)
        trainer.run(__C.run_mode)
