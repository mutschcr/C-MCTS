# C-MCTS: Safe Planning with Monte Carlo Tree Search

This repository contains the implementation of Constrained Monte Carlo Tree Search (C-MCTS). The MCTS algorithm has been implemented using Python, while the environment is a C++ implementation integrated into the Python code using pybind.

## Reference

Please cite our work if you find our work useful for your research:
```
@article{parthasarathy2023,  
    title={C-MCTS: Safe Planning with Monte Carlo Tree Search},  
    author={Parthasaraty, Dinesh and Kontes, Georgios and Plinge, Axel and Mutschler, Christopher},  
    journal={arXiv preprint arXiv:XXXX.XXXXX},  
    year={2023}  
}
```

## Installation

To set up a conda environment, follow these steps:

1. Create a conda environment using the provided `environment.yml` file:
   ```bash
   conda env create --name cmcts-env --file environment.yml
2. Activate the created conda environment:
   ```bash
   conda activate cmcts-env
   
 ## Usage
 
To run the code, use the following command:

  `python run.py <environment> <mode> <planning_iterations> <ensemble_threshold> <alpha_0> <epsilon> <cost_constraint> <maximum_training_loops>`

Replace the placeholders with the desired values for the parameters:

- `<environment>`: The environment to run the algorithm on. Choose one of the following:
  - `rocksample_5_7`: Rocksample environment of size 5x5 with 7 rocks.
  - `rocksample_7_8`: Rocksample environment of size 7x7 with 8 rocks.
  - `rocksample_11_11`: Rocksample environment of size 11x11 with 11 rocks.
  - `safegridworld`: SafeGridWorld environment.
- `<mode>`: The mode of operation. Choose either "train" or "evaluate".
- `<planning_iterations>`: The number of MCTS iterations per time step.
- `<ensemble_threshold>`: The standard deviation threshold to detect out-of-distribution inputs.
- `<alpha_0>`: The initial step size to update the Lagrange multiplier during training.
- `<epsilon>`: The termination criterion for the training loop.
- `<cost_constraint>`: To limit the costs incurred by the agent in an episode.
- `<maximum_training_loops>`: The maximum number of training loops (upper limit) to train the safety critic.
