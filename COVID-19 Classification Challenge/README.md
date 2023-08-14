## COVID-19 CLASSIFICATION CHALLENGE

## Running the files: ##

1. Open this folder in jupyter notebook.

1. Training & Testing Densenet121
	1. Run `densenet.py` and generate `.h5` file
	1. Run `densenet_extra_train.py`. 2 H5 files will be present. IMPORTANT: Delete `trained_densenet_model.h5` and then rename
	    `trained_densenet_model2.h5` as `trained_densenet_model.h5`.
	1. Run `densenet_run.py` and see its performance.

1. Training & Testing InceptionV3
	1. Run `inception.py` and generate `.h5` file
	1. Run `inception_extra_train.py`. 2 H5 files will be present. IMPORTANT: Delete `trained_inception_model.h5` and then rename
	    `trained_inception_model2.h5` as `trained_inception_model.h5`.
	1. Run `inception_run.py` and see its performance.

1. Ensemble of above 2 models
	Ensure that `trained_densenet_model.h5` and `trained_inception_model.h5` files are present. Then run `combination.py` and see the
	performance.

## Note: ##
The `images` directory contains screenshots of the results of each model.

The datasets can be accessed [here](https://drive.google.com/drive/folders/1pSHVwZQvvGrJjcxgszOMCGfM65DE4ZL1?usp=sharing).
