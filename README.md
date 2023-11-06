# Rainbow DQN

Implementation of [Rainbow-DQN](https://arxiv.org/abs/1710.02298) agent for state-of-the-art discrete action design space exploration.

### Installation

Execute:

```powershell
conda env create --file environment.yml
conda activate rainbow
```

Optional:

Upgrade [PyTorch](https://pytorch.org/):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### NOTES

Tested on [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) environment.

### TODO

- [ ] Complete comments and docstrings
- [ ] Add flexibility in environment for either 1D or 2D observations
- [ ] Enrich this README (main page) with a description of the project, results, lab page, etc.
