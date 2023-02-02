To set up the environment:
- Ensure the following are installed:
    - Python 3.10
    - CUDA toolkit and cudnn
        - The code can be run without a GPU, by modifying the JAX dependency in `pyproject.toml`
    - [Poetry](https://python-poetry.org/)
- Install the environment: `poetry install`
- Activate the environment: `poetry shell`

To run the experiments:
- Composite testing experiment (Table 2): `python src/experiments/gandk_test.py --repeats 0 150`
- Examination of g and k parameter estimates (Figure 5): `python src/experiments/gandk_opt.py`

Development tools:
- Typechecker: `mypy --ignore-missing-imports src tests`
- Unit tests: `pytest`
