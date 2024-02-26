# Kelp Wanted: Segmenting Kelp Forests

This is Team Epoch's solution to
the [Kelp Wanted: Segmenting Kelp Forests](https://www.drivendata.org/competitions/255/kelp-forest-segmentation/)
competition.

A [technical report](Detect_Kelp___Technical_Report.pdf) is included in this repository.

## Getting started

This section contains the steps that need to be taken to get started with our project and fully reproduce our best
submission on the private leaderboard. The project was developed on Windows 10/11 OS on Python 3.10.13 on Pip version 23.2.1.

### 0. Prerequisites
Models were trained on machines with the following specifications:
- CPU: AMD Ryzen Threadripper Pro 3945WX 12-Core Processor / AMD Ryzen 9 7950X 16-Core Processor
- GPU: NVIDIA RTX A5000 / NVIDIA RTX Quadro 6000 / NVIDIA RTX A6000
- RAM: 96GB / 128GB
- OS: Windows 10/11
- Python: 3.10.13
- Estimated training time: 3-6 hours per model on these machines.

For running inference, a machine with at least 32GB of RAM is recommended. We have not tried running the inference on a
machine with less RAM using all the test data that was provided by DrivenData.

### 1. Clone the repository

Make sure to clone the repository with your favourite git client or using the following command:

```
https://github.com/TeamEpochGithub/iv-q2-detect-kelp.git
```

### 2. Install Python 3.10.13

You can install the required python version here: [Python 3.10.13](https://github.com/adang1345/PythonWindows/blob/master/3.10.13/python-3.10.13-amd64-full.exe)

### 3. Install the required packages

Install the required packages (on a virtual environment is recommended) using the following command:
A .venv would take around 7GB of disk space.

```shell
pip install -r requirements.txt
```

### 4. Setup the competition data

The data of the competition can be downloaded here: [Kelp Wanted: Segmenting Kelp Forests](https://www.drivendata.org/competitions/255/kelp-forest-segmentation/data/)
Unzip all the files into the `data/raw` directory.
The structure should look like this:

```
data/
    ├── processed/ # Cache directory for processed data. 
    ├── raw/ # @TODO Competition data must be PLACED HERE
        ├── submission_format/ # @TODO Submission format from the competition
        ├── test_satellite/ # @TODO Test satellite images from the competition
        ├── train_kelp/ # @TODO Train kelp images from the competition
        ├── train_satellite/ # @TODO Train satellite images from the competition
        ├── metadata_fTq0l2T.csv # @TODO Metadata from the competition
    ├── test/ # Test caching
    ├── training/ # Training cache
```

### 5. Main files explanation

- `train.py`: This file is used to train a model. `train.py` reads a configuration file from `conf/train.yaml`. This configuration file 
contains the model configuration to train with additional training parameters such as test_size and a scorer to use. 
The model selected in the `conf/train.yaml` can be found in the `conf/model` folder where a whole model configuration is stored (from preprocessing to postprocessing).
When training is finished, the model is saved in the `tm` directory with a hash that depends on the specific pre-processing, pretraining steps + the model configurations.

    - Command line arguments
    - CUDA_VISIBLE_DEVICES: The GPU to use for training. If not specified it uses DataParallel to train on multiple GPUs.  If you have multiple GPUs, you can specify which one to use.
- `submit.py`: This file does inference on the test data from the competition given trained model or an ensemble of trained models. 
It reads a configuration file from `conf/submit.yaml` which contains the model/ensemble configuration to use for inference.
Model configs can be found in the `conf/model` folder and ensemble configs in the `conf/ensemble` folder. The `conf/ensemble`
folder specifies the models (`conf/model`) to use for the ensemble and the weights to use for each model. The `submit.py` 

### 6. Place the fitted models
(For DrivenData) Any additional supplied trained models /scalers (.pt / .gbdt / .scaler) should be placed in the `tm` directory. 
When these models were trained, they are saved with a hash that depends on the specific pre-processing, pretraining steps + the model configurations.
In this way, we ensure that we load the correct saved model automatically when running `submit.py`.

### 7. Run submit.py

For reproducing our best submission, run `submit.py`. This will load the already configured `submit.yaml` file and
run the inference on the test data from the competition. `submit.yaml` in configured to what whe think is our best and our
most robust solution:
![img.png](images/best_submission.png)
`conf/ensemble/21-02-1st.yaml` contains a mixed ensemble of models trained on different feature sets, including UNets with VGG encoders, SwinTransformers and a ConvNext model.


If you get an error of that the path was not found of a model. Please ensure that you have the correct trained model in the `tm` directory.
If you don't have the trained models, you can train them 1 by 1 using `train.py` and the `conf/train.yaml` file.

`submit.py` will output a `submission.zip` in the root of the project directory. This file can be uploaded to the competition page. 


## Quality Checks

Quality checks are performed using [pre-commit](https://pre-commit.com/) hooks. To install these hooks, run:

```shell
pre-commit install
```

To run the pre-commit hooks locally, do:

```shell
pre-commit run --all-files
```

## GBDT

Due to long training times, we reused trained models for the GBDT. To train these, set `test_size` to zero, and remove
the `saved_at` from a model config.
Once it is saved, it is possible to set `saved-at` to the filename of the saved model, and use this for any runs
regardless of test size.



## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/).

To make the documentation, run `make html` with `docs` as the working directory. The documentation can then be found in `docs/_build/html/index.html`.

Here's a short command to make the documentation and open it in the browser:

```shell
cd ./docs/;
./make.bat html; start chrome file://$PWD/_build/html/index.html
cd ../
```
