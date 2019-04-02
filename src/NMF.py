from functools import partial
import argparse
import json
import glob
import os

import numpy as np

from extraction import NMF
import imageio
import joblib


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


def NMF_helper(datafile, outpath, save_individual, nmf_args):
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
    """
    im_files = sorted(glob.glob(datafile+"/*"))
    data = np.array([imageio.imread(f) for f in im_files])
    key = datafile.split(os.sep)[-1]
    fname = "{}.jason".format(key)
    outfile = os.path.join(outpath, fname)
    regs = fit_NMF(data, n_comps=nmf_args[0], iters=nmf_args[1],
                   percentile=nmf_args[2], chunk_size=nmf_args[3],
                   overlap=nmf_args[4])
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
        (d, args['output'], args['save_individual'], nmf_args)
        for d in datapaths
    )

    if not args['save_individual']:
        fname = "neurofinder_nmf.json"
        outfile = os.path.join(args['output'], fname)
        output = [i for i in out]
        with open(outfile, 'w') as outF:
            json.dump(output, outF)
