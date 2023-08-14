## COVID-19 Classification Challenge

**Task**: Image Classification

**Data**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

Training data has 2318 samples.
1. Label 0 is assigned for COVID-19 cases.
2. Label 1 is assigned for normal cases.
3. Label 2 is assigned for viral pneumonia cases.

The datasets can be accessed [here](https://drive.google.com/drive/folders/1pSHVwZQvvGrJjcxgszOMCGfM65DE4ZL1?usp=sharing).

## Running the files:

1. Open this folder in Jupyter Notebook.

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

1. Ensemble of the two models

	Ensure that `trained_densenet_model.h5` and `trained_inception_model.h5` files are present. Then run `combination.py` and see the
	performance.

## Note:
The `images` directory contains screenshots of the results of each model.
