import argparse
import os
import glob
from functools import partial
import imageio
import json

import joblib
import numpy as np
from extraction import NMF


def fit_NMF(data, n_comps=3, iters=50, percentile=95, chunk_size=(60, 60),
            overlap=0.1):

    model = NMF(k=n_comps, max_iter=iters, percentile=percentile,
                overlap=overlap)
    model = model.fit(data, chunk_size=chunk_size, padding=(20, 20))
    merged = model.merge(overlap=overlap, max_iter=iters, k_nearest=20)
    regions = [{'coordinates': region.coordinates.tolist()} for
               region in merged.regions]

    return regions


def output_json(regions, outfile):
    with open(outfile, 'w') as outF:
        json.dump(regions, outF)


def NMF_helper(datafile, outpath):
    im_files = sorted(glob.glob(datafile+"/*"))
    data = np.array([imageio.imread(f) for f in im_files])
    key = datafile.split(os.sep)[-1]
    fname = "{}.json".format(key)
    outfile = os.path.join(outpath, fname)
    regs = fit_NMF(data)
    output_json(regs, outfile)


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

    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    files = os.listdir(args['input'])
    pref = partial(os.path.join, args['input'])
    datapaths = list(map(pref, files))
    print(datapaths)

    # find NMF in parallel with joblib
    out = joblib.Parallel(n_jobs=args['n_jobs'], verbose=10)(
        joblib.delayed(NMF_helper)
            (d, args['output'])
        for d in datapaths
    )
