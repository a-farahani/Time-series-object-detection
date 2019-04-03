# team-rhodes-P3
[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)


## Table of contents

1. [Purpose](#purpose)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [License](#license)
6. [Contact info](#contact-info)

## Purpose

The goal of team-rhodes-p3 is to detect neurons in images (in the format of time-series). This project is intended to be the solution for project 3, course CSCI8360 at the University of Georgia. 

## Requiremnets

This project requires Python 3.x with `cv2`, `imageio`, `thunder-extraction`, `joblib`, and `image_slicer`  libraries installed.

## Dataset

The dataset consists of a set of 28 samples (19 training samples and 9 test samples). Eeach sample is a a set of `.tiff` images grouped into a directory.  

Since the size of the dataset is large, the data is not included in the repository, however a script to download the dataset is provided in the repository. To download the data navigate to `scripts/` directory and run the following in command line (note that [`google-cloud-sdk`](https://cloud.google.com/sdk/) is required for the script): 

`$ ./get_files.sh`

## Usage

You may download the source code and simply run the following command:

`$ python3 NMF.py -i 'path/to/images/'`

List of command line arguments to pass to the program are as follows:

	--input: Path to image sets.
	--output: Path to save json files to.
	--n_jobs: number of jobs to spawn in parallel
	--save_individual: if set each dataset will be saved separately.
	--n_comps: number of components to estimate per block.
	--iters: max number of algorithm iterations.
	--perc: the value for thresholding (higher is more thresholding).
	--chunk_size: width and height of chunk, two values.
	--overlap: value determining whether to merge.
	--custom_config: if true use the best configuration.

The see the above list in command line execute the following command:

`$ python3 NMF.py -h`

Alternatively, you can install the package using `pip` as follows:

`$ pip install --user -i https://test.pypi.org/simple/rhodes`

In this case you can use the other functionality of this package to augment and prepare the dataset for Tiramisu model that can be obtained from [here](https://github.com/dsp-uga/team-linden-p2):

## License
The code in this repository is free software: you can redistribute it and/or modify it under the terms of the MIT lisense. 

## Contact info

For questions please email one of the authors: 

**durden4th@gmail.com**

**a.farahani@uga.edu**

**saedr@uga.edu**
