# Prompt-Optimizer

## Overview
The prompt-optimizer project is designed to iteratively optimize prompts for Large Language Models (LLMs), specifically for the task of smart contract analysis. It employs an evolutionary algorithm-inspired approach to discover effective prompts that yield better results from an LLM based on predefined fitness criteria. The core idea is to refine prompts over generations by evaluating their performance on a given dataset and selecting the best-performing ones for further mutation and evolution. This system aims to automate the process of prompt engineering, making LLM applications more robust and efficient by finding optimal instructions for specific tasks.

## Deployment Guide

### Quick Start

1. **Application Setup**:
```bash
cd prompt-optimizer
python3 -m venv venv
`source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
cp .env.template .env
pip install -r requirements.txt
```

2. **Environment Configuration**:
```bash
# Edit .env with production settings
nano .env
```

3. **Run the Optimization**:
Execute `main.py` to run the prompt optimization process. This will generate the `evolution_history.json` file in the `result` directory.

4. **Generate the Plot**:
After `main.py` has completed, run `plot_evolution.py` to generate the visualization.



Production `.env` example:
```
OPENAI_API_KEY=your-openai-key
```

## Documentation and Configuration

### 1. **Configuration Parameters (config.py)**

This design choice significantly improves the usability and maintainability of the project, allowing users to easily adjust the behavior
of the optimizer without delving into the core logic.
The parameters are well-commented and self-explanatory, covering various aspects of the evolutionary algorithm:

- `T_max (Max Generations)`: Maximum number of generations. This acts as a hard stop to prevent infinite loops and
control computational resources.
- `P_S (Population Size)`:  Specifies the number of prompts maintained in the population at each generation.
- `k_e (Elite Count)`: Determines how many of the best-performing prompts from the current generation are directly carried over to the next. 
- `m (Replay Buffer Capacity)`:  Sets the maximum number of top-performing prompts stored in the replay buffer. The replay buffer helps in retaining historical good prompts and influencing the fitness calculation of current prompts.
- `tau_max (Max Mutation Temperature)`: Controls the initial strength of mutations applied to prompts. A higher value means more drastic changes, encouraging exploration in early stages.
- `beta (Mutation Temperature Decay Rate)`:  Dictates how quickly the mutation strength decreases over generations. This allows for broader exploration initially and finer-grained refinement as the optimization progresses.
- `delta_fitness (Min Fitness Improvement)`: A threshold used to determine if a generation has made significant progress. If the improvement in best fitness is below this value, it contributes to the `N_stable` count.
- `N_stable (Number of Stable Generations)`: : If the best fitness does not significantly improve for this many consecutive generations, the optimization process is considered to have converged and stops.
- `D_min (Minimum Population Diversity)`: A threshold for population diversity. If the diversity falls below this value, it indicates that the population has become too homogeneous, and the optimization stops to prevent premature convergence to a suboptimal solution.
- `w_exec (Weight for Execution Score)`: The weight assigned to the execution score component of fitness. This allows for adjusting the importance of execution accuracy in the overall fitness evaluation.
- `w_log (Weight for Log/Auxiliary Score)`: The weight assigned to the log/auxiliary score component of fitness. This allows for adjusting the importance of auxiliary information in the overall fitness evaluation.
- `epsilon (Weight for Replay Buffer's Contribution to Fitness)`: The weight assigned to the replay buffer's contribution to fitness. This allows for adjusting the influence of historical good prompts on the current generation's fitness.
- `alpha (Smoothing Factor for EMA of Fitness)`: The smoothing factor used in the Exponential Moving Average (EMA) calculation of fitness. This helps in smoothing out the fitness values and providing a more stable evaluation.
- `lambda_complexity (Regularization Parameter for Prompt Complexity)`: A regularization parameter that penalizes overly complex prompts. This helps in maintaining a balance between prompt complexity and effectiveness.
- `w_cov (Weight for Component Coverage in f_exec)`: The weight assigned to the component coverage component of fitness in the execution score. This allows for adjusting the importance of component coverage in the execution score.
- `w_det (Weight for Detail Accuracy in f_exec)`: The weight assigned to the detail accuracy component of fitness in the execution score. This allows for adjusting the importance of detail accuracy in the execution score.
- `w_F (Weight for Function F1 score in f_coverage)`: The weight assigned to the Function F1 score component of fitness in the coverage score. This allows for adjusting the importance of Function F1 score in the coverage score.
- `seed_prompts (Initial Prompts)`: A list of initial prompts that are used to seed the first generation of the optimization process. These prompts serve as a starting point for the evolutionary algorithm.

### 2. Data Files 

The data directory contains two crucial JSON files: `train_data.json` and `val_data.json`. These files serve as the ground truth for evaluating the performance
of the LLM with different prompts.