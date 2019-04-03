from functools import partial
import argparse
import json
import glob
import os

import numpy as np
from extraction import NMF
import imageio
import joblib
import cv2


def fit_NMF(data, n_comps=3, iters=50, percentile=95, chunk_size=(60, 60),
            overlap=0.1):
    """
    fits nmf to dataset

    Parameters
    ----------
    data : numpy matrix
        the video to which the NMF is fit
    n_comps : int
        number of components to estimate per block
    iters : int
        max number of algorithm iterations
    percentile : int
        the value for thresholding
    chunk_size : tuple
        width and height of chunk, two values
    overlap : float
        value determining whether to merge

    Returns
    -------
    regions : list
        a list of regions extracted by NMF
    """

    model = NMF(k=n_comps, max_iter=iters, percentile=percentile,
                overlap=overlap)
    model = model.fit(data, chunk_size=chunk_size, padding=(20, 20))
    merged = model.merge(overlap=overlap, max_iter=iters, k_nearest=20)
    regions = [{'coordinates': region.coordinates.tolist()} for
               region in merged.regions]

    return regions


def output_json(regions, outfile):
    """
    used to write an individual file's region to JSON

    Parameters
    ----------
    regions : list
        raw output of fit_NMF
    outfile : string
        filename to save regions to as JSON
    """
    with open(outfile, 'w') as outF:
        json.dump(regions, outF)


def config(key):
    if '00.00' in key:
        conf = (10, 20, 95, (50, 50), 0.1)
    elif '00.01' in key or '01.00' in key:
        conf = (5, 30, 95, (50, 50), 0.1)
    elif '01.01' in key:
        conf = (3, 50, 95, (50, 50), 0.1)
    elif '02.00' in key or '02.01' in key:
        conf = (5, 50, 99, (100, 100), 0.1)
    elif '03.00' in key:
        conf = (10, 30, 95, (50, 50), 0.1)
    elif '04.00' in key:
        conf = (5, 50, 99, (50, 50), 0.1)
    elif '04.01' in key:
        conf = (3, 50, 95, (60, 60), 0.1)
    return conf


def NMF_helper(datafile, outpath, save_individual, nmf_args, custom_config):
    """
    helper function to load data and run fit_NMF

    Parameters
    ----------
    datafile : string
        file directory containing multiple datsets
    outpath : string
        directory to save output
    save_individual : bool
        if true calls output_json, else simply returns dictionary
    nmf_args : list
        list of each parameter for the NMF model
    custom_config : bool
        if true use best configuration, else use nmf_args.
    """
    im_files = sorted(glob.glob(datafile + "/images/*"))
    data = np.array([cv2.imread(f, 0) for f in im_files])
    data = cv2.normalize(data, None, alpha=0, beta=1,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    key = datafile.split(os.sep)[-1]
    fname = "{}.json".format(key)
    outfile = os.path.join(outpath, fname)
    regs = fit_NMF(data, n_comps=nmf_args[0], iters=nmf_args[1],
                   percentile=nmf_args[2], chunk_size=nmf_args[3],
                   overlap=nmf_args[4])

    if custom_config:
        conf = config(key)
        regs = fit_NMF(data, conf[0], conf[1], conf[2], conf[3], conf[4])
    if save_individual:
        output_json(regs, outfile)
    return {"dataset": key, "regions": regs}


if __name__ == '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description='reads image sets and computes NMF based regions',
        prog='NFM.py <args>')

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help="Path to image sets")

    # Optional arguments
    parser.add_argument("-o", "--output", default=os.path.join(cwd, "JSON"),
                        help="Path to save json files to")
    parser.add_argument("-n", "--n_jobs", type=int, default=-1,
                        help="number of jobs to spawn in parallel")
    parser.add_argument("-d", "--save_individual", action="store_true",
                        help="if set each dataset will be saved separately")
    parser.add_argument("--n_comps", type=int, default=3,
                        help="number of components to estimate per block")
    parser.add_argument("--iters", type=int, default=50,
                        help="max number of algorithm iterations")
    parser.add_argument("--perc", type=int, default=95,
                        help="the value for thresholding"
                             " (higher is more thresholding)")
    parser.add_argument("--chunk_size", nargs=2, type=int, default=[60, 60],
                        help="width and height of chunk, two values")
    parser.add_argument("--overlap", type=float, default=0.1,
                        help="value determining whether to merge")
    parser.add_argument("--custom_config", type=bool, default=False,
                        help="if true use best configuration")
    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    # parse NMF args
    chunk_size = tuple(args['chunk_size'])
    nmf_args = [args['n_comps'], args['iters'], args['perc'],
                chunk_size, args['overlap']]

    files = os.listdir(args['input'])
    pref = partial(os.path.join, args['input'])
    datapaths = list(map(pref, files))
    print(datapaths)

    # find NMF in parallel with joblib
    out = joblib.Parallel(n_jobs=args['n_jobs'], verbose=10)(
        joblib.delayed(NMF_helper)
        (d, args['output'], args['save_individual'], nmf_args, args['custom_config'])
        for d in datapaths
    )

    if not args['save_individual']:
        fname = "neurofinder_nmf.json"
        outfile = os.path.join(args['output'], fname)
        output = [i for i in out]
        with open(outfile, 'w') as outF:
            json.dump(output, outF)
