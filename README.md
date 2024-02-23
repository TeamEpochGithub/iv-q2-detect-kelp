# Kelp Wanted: Segmenting Kelp Forests

This is Team Epoch's solution to
the [Kelp Wanted: Segmenting Kelp Forests](https://www.drivendata.org/competitions/255/kelp-forest-segmentation/)
competition.

A [technical report](Detect_Kelp___Technical_Report.pdf) is included in this repository.

## Getting started

This section contains the steps that need to be taken to get started with our project and fully reproduce our best
submission on the private leaderboard. The project was developed on Windows 10/11 OS on Python 3.10.13.

### 1. Clone the repository

Make sure to clone the repository with your favourite git client or using the following command:

```shell
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
Unzip all the files into the `data` directory.
The structure should look like this:

```shell
data/
    ├── test_images/
    ├── train_images/
    ├── train_masks/
    ├── sample_submission.csv
    ├── train.csv
    ├── test.csv
```

### 4. Run submit.py



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

```shell

## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/).

To make the documentation, run `make html` with `docs` as the working directory. The documentation can then be found in `docs/_build/html/index.html`.

Here's a short command to make the documentation and open it in the browser:

```shell
cd ./docs/;
./make.bat html; start chrome file://$PWD/_build/html/index.html
cd ../
```
