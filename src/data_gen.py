import argparse
import sys
import os

from util import combine, json_to_mask, slicer, rotate, transpose, split


def main():
    """
    main function to prepare data for Tiramisu algorithm
    """
    parser = argparse.ArgumentParser(
        description='reads image sets and augments the data for Tiramisu',
        prog='data_gen.py <args>')

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help="Path to image sets")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to save test and train files")

    # Optional arguments
    parser.add_argument("-r", "--ratio", type=float, default=0.2,
                        help="validation set ratio")

    # Creating required directories
    args = vars(parser.parse_args())
    if not os.path.exists(args['output'] + '/train/data/'):
        os.makedirs(args['output'] + '/train/data/')
    if not os.path.exists(args['output'] + '/validate/data/'):
        os.makedirs(args['output'] + '/validate/data/')
    if not os.path.exists(args['output'] + '/train/masks/'):
        os.makedirs(args['output'] + '/train/masks/')
    if not os.path.exists(args['output'] + '/validate/masks/'):
        os.makedirs(args['output'] + '/validate/masks/')
    if not os.path.exists(args['output'] + '/test/data/'):
        os.makedirs(args['output'] + '/test/data/')

    print("Creating an image per video...")
    combine(args['input'], args['output'])

    print("Generating a mask per video...")
    json_to_mask(args['input'], args['output'])

    print("augmenting the dataset...")
    slicer(args['output'])
    rotate(args['output'])
    transpose(args['output'])

    # Splitting the dataset into training and validation set
    split(args['output'], args['ratio'])


if __name__ == "__main__":
    sys.exit(main())
