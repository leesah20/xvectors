import argparse
import json
import sys

import torch
from training import Trainer


class Arg(object):
    pass


def main(argv: Arg):
    config = json.load(argv.config_file)

    trainer = Trainer.XVectorTrainer(
        config,
        ckpt_path=argv.checkpoint_path,
        language=argv.language,
        gender=argv.gender
    )

    trainer.build()

    if argv.load_from_epoch:
        trainer.load_checkpoint(argv.load_from_epoch)

    if argv.action in ['all', 'train']:
        trainer.train()

    if argv.action in ['all', 'evaluate']:
        trainer.evaluate()

    if argv.action in ['all', 'extract']:
        trainer.save_features(argv.save_path, argv.extract_language, projection=argv.projection)


def check_args():
    argv = Arg()
    arg_parser = argparse.ArgumentParser(argument_default=None)
    arg_parser.add_argument('action', choices=['all', 'train', 'evaluate', 'extract'])
    arg_parser.add_argument('config_file', type=argparse.FileType('r'))
    arg_parser.add_argument('checkpoint_path', type=str)
    arg_parser.add_argument('--load_from_epoch', default=0, type=int)
    arg_parser.add_argument('--language', default="english_full", choices=["english_full", "xitsonga"])
    arg_parser.add_argument('--gender', action='store_true')
    extract_mode = 'all' in sys.argv or 'extract' in sys.argv
    arg_parser.add_argument('--save_path', required=extract_mode, type=str)
    arg_parser.add_argument('--extract_language', default="english_full", choices=["english_full", "xitsonga"])
    arg_parser.add_argument('--projection', default='lda', choices=['lda', 'mean', 'pca'])
    arg_parser.parse_args(namespace=argv)
    return argv

if __name__ == '__main__':
    arg = check_args()
    main(arg)
