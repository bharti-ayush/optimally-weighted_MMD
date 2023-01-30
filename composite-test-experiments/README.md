To run the experiments:
- Ensure Python 3.10 is available (e.g. using conda or pyenv), and Poetry is installed
- Run `poetry install`
- Composite testing experiment (Table 2): `python src/experiments/weighted_mmd/gandk_test.py --repeats 0 150`
- Examination of g and k parameter estimates (Figure 5): `python src/experiments/weighted_mmd/gandk_opt.py`
