# prompt_optimizer_project/apps/config.py

OPTIMIZER_CONFIG = {
    "T_max": 10,             # Max generations
    "P_S": 20,               # Population size
    "k_e": 10,               # Elite count (number of best individuals to carry over)
    "m": 10,                 # Replay buffer capacity (stores top-m performers, using k_e as m in the text)
    "tau_max": 0.7,         # Max mutation temperature (controls initial mutation strength)
    "beta": 0.1,            # Mutation temperature decay rate (how quickly mutation strength decreases)
    "delta_fitness": 0.005, # Min fitness improvement to be considered significant
    "N_stable": 5,          # Number of stable generations (no significant improvement) before stopping
    "D_min": 0.3,           # Minimum population diversity (for convergence check)
    "w_exec": 0.7,          # Weight for the execution score component of fitness
    "w_log": 0.3,           # Weight for the log/auxiliary score component of fitness
    "epsilon": 0.1,         # Weight for the replay buffer's contribution to fitness
    "alpha": 0.3,           # Smoothing factor for Exponential Moving Average (EMA) of fitness
    "lambda_complexity": 0.01, # Regularization parameter for prompt complexity
    "w_cov": 0.6,           # Weight for Component Coverage in f_exec
    "w_det": 0.4,           # Weight for Detail Accuracy in f_exec
    "w_F": 0.25,            # Weight for Function F1 score in f_coverage
    "w_V": 0.25,            # Weight for State Variable F1 score in f_coverage
    "w_M": 0.25,            # Weight for Modifier F1 score in f_coverage
    "w_E": 0.25,            # Weight for Event F1 score in f_coverage
    "seed_prompts": [       # Initial prompts to seed the first generation
        "Analyze the provided smart contract. Focus on its function definitions, state variables, modifiers, and events. Be precise.",
        "List all functions, state variables, modifiers, and events in the smart contract. For each function, detail its parameters, return values, and purpose. For state variables, list type and purpose. For modifiers, explain their checks. For events, list parameters and when they are emitted."
    ]
}


