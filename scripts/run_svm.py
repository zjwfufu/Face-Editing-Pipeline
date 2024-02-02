"""code is adapted from https://github.com/genforce/interfacegan/tree/master"""

import os
import numpy as np
import argparse

import sys
cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_path, '../'))


from common.svm_util import train_boundary

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train semantic boundary with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-o', '--output_dir', type=str, default='./scripts/bound',
                      help='Directory to save the output results. (required)')
  parser.add_argument('-c', '--latent_codes_path', type=str, default='./scripts/gen_w/w.npy',
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=True,
                      help='Path to the input attribute scores. (required)')
  parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.2,
                      help='How many samples to choose for training. '
                           '(default: 0.2)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
  parser.add_argument('-V', '--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')

  return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    print('Loading latent codes.')
    if not os.path.isfile(args.latent_codes_path):
        raise ValueError(f'Latent codes `{args.latent_codes_path}` does not exist!')
    latent_codes = np.load(args.latent_codes_path)

    print('Loading attribute scores.')
    if not os.path.isfile(args.scores_path):
        raise ValueError(f'Attribute scores `{args.scores_path}` does not exist!')
    scores = np.load(args.scores_path)

    print('check scores stat:')
    print('max:   ', np.max(scores))
    print('min:   ', np.min(scores))
    print('mean:  ', np.mean(scores))
    print('std:   ', np.std(scores))
    print('shape:   ', scores.shape)

    boundary = train_boundary(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=args.chosen_num_or_ratio,
                            split_ratio=args.split_ratio,
                            invalid_value=args.invalid_value)
  
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")

    np.save(os.path.join(args.output_dir, 'boundary.npy'), boundary)


if __name__ == '__main__':
    main()

