# Data Science Repo Template

## How to use:
Install Cookiecutter:
```
pip install cookiecutter
```
Create a project based on the template:
```
cookiecutter https://github.com/yeungadrian/DSProjectTemplate.git
```

## Tools used in this project
* [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/index.html): Manage configuration
* [Poetry](https://python-poetry.org/docs/basic-usage/): Dependency management
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pytest](https://docs.pytest.org/en/latest/): Write small, readable tests
* [python-dotenv](https://pypi.org/project/python-dotenv/): Local testing of secrets

## Project structure
```
.
├── config
│   ├── main.yaml               # Main configuration file
├── datasets                    # store data locally
│   ├── external                # data from third party sources.
│   ├── interim                 # intermediate data that has been transformed
│   ├── processed               # the final, canonical data sets for modeling
│   └── raw                     # the original, immutable data dump.
├── notebooks                   # store notebooks
├── src                         # store source code
│   ├── __init__.py             # make src a Python module
│   ├── data_ingestion          # data ingestion pipeline & classes
│   ├── feature_engineering     # feature engineering pipeline & classes
│   └── modelling               # modelling pipeline & classes
├── tests                       # store tests
│   └── __init__.py             # make tests a Python module
├── .gitignore                  # ignore files that cannot commit to Git
├── .pre-commit-config.yaml     # configurations for pre-commit
├── Makefile                    # store useful commands
├── pyproject.toml              # dependencies for poetry
├── README.md                   # describe your project
└── setup.cfg                   # flake8 config
```
