# Orthogonality Informed Adaptive Low Rank Training

# Description
Orthogonal DPNN training methods

# Quickstart

## Create the pipeline environment and install the madonna package
Before using the template, one needs to install the project as a package.
* First, create a virtual environment.
> You can either do it with conda (preferred) or venv.
* Then, activate the environment
* Finally, install the project as a package. Run:
```
pip install -e .
```
## Run the MNIST example
This pipeline comes with a toy example (MNIST dataset with a simple feedforward neural network). To run the training (resp. testing) pipeline, simply run:
```
python scripts/train.py
# or python scripts/test.py
```
Or, if you want to submit the training job to a submit (resp. interactive) cluster node via slurm, run:
```
sbatch job_submission.sbatch
# or sbatch job_submission_interactive.sbatch
```
> * The experiments, evaluations, etc., are stored under the `logs` directory.


# Project Organization
```
├── configs                              <- Hydra configuration files
│   ├── callbacks                               <- Callbacks configs
│   ├── data                                    <- Datamodule configs
│   ├── debug                                   <- Debugging configs
│   ├── experiment                              <- Experiment configs
│   ├── local                                   <- Local configs
│   ├── log_dir                                 <- Logging directory configs
│   ├── logger                                  <- Logger configs
│   ├── model                                   <- Model configs
│   ├── trainer                                 <- Trainer configs
│   │
│   ├── test.yaml                               <- Main config for testing
│   └── train.yaml                              <- Main config for training
│
├── docs                                 <- Directory for Sphinx documentation in rst or md.
├── models                               <- Trained and serialized models, model predictions
├── notebooks                            <- Jupyter notebooks.
├── reports                              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                                 <- Generated plots and figures for reports.
├── scripts                              <- Scripts used in project
│   ├── job_submission.sbatch               <- Submit training job to slurm
│   ├── job_submission_interactive.sbatch   <- Submit training job to slurm (interactive node)
│   ├── test.py                             <- Run testing
│   └── train.py                            <- Run training
│
├── src/madonna             <- Source code
│   ├── datamodules                             <- Lightning datamodules
│   ├── models                                  <- Lightning models
│   └── utils                                   <- Utility scripts
│
├── .gitignore                           <- List of files/folders ignored by git
├── .pre-commit-config.yaml              <- Configuration of pre-commit hooks for code formatting
├── requirements.txt                     <- File for installing python dependencies
├── setup.cfg                            <- Configuration of linters and pytest
├── LICENSE.txt                          <- License as chosen on the command-line.
├── pyproject.toml                       <- Build configuration. Don't change! Use `pip install -e .`
│                                           to install for development or to build `tox -e build`.
├── setup.cfg                            <- Declarative configuration of your project.
├── setup.py                             <- [DEPRECATED] Use `python setup.py develop` to install for
│                                           development or `python setup.py bdist_wheel` to build.
└── README.md
```
