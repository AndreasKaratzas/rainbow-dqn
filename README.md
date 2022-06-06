# Rainbow DQN

[//]: # (There is a bug related to PyLint. The `src` directory is not properly recognized and therefore PyLint cannot find the imported files throughout the project. To temporarily resolve this issue, set "python.analysis.autoSearchPaths": false .)

[//]: # (1. pip install msgpack-rpc-python 2. pip install airsim 3. pip install -e envs)

### Installation

Execute:

```powershell
conda env create --file environment.yml
conda activate rainbow
```

### TODO

- [ ] Add non-learning episodes in the beginning for warmup
- [ ] Add `yaml` configuration file utilities as well as argument parser
- [ ] Complete comments and docstrings
- [ ] Add flexibility in environment for either 1D or 2D observations
