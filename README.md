# Kelp Wanted: Segmenting Kelp Forests

This is Team Epoch's solution to the [Kelp Wanted: Segmenting Kelp Forests](https://www.drivendata.org/competitions/255/kelp-forest-segmentation/) competition.

A [technical report](Detect_Kelp___Technical_Report.pdf) is included in this repository.

## Getting started

Install the required packages using:

```shell
pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html
```

## Quality Checks

Quality checks are performed using [pre-commit](https://pre-commit.com/) hooks. To install these hooks, run:

```shell
pre-commit install
```

To run the pre-commit hooks locally, do:

```shell
pre-commit run --all-files
```

## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/).

To make the documentation, run `make html` with `docs` as the working directory. The documentation can then be found in `docs/_build/html/index.html`.

Here's a short command to make the documentation and open it in the browser:

```shell
cd ./docs/;
./make.bat html; start chrome file://$PWD/_build/html/index.html
cd ../
```
