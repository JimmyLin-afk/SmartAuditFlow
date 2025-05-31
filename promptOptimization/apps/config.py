# prompt_optimizer_project/apps/config.py

OPTIMIZER_CONFIG = {
    "T_max": 10,             # Max generations
    "P_S": 6,               # Population size
    "k_e": 2,               # Elite count (number of best individuals to carry over)
    "m": 5,                 # Replay buffer capacity (stores top-m performers)
    "tau_max": 0.3,         # Max mutation temperature (controls initial mutation strength)
    "beta": 0.1,            # Mutation temperature decay rate (how quickly mutation strength decreases)
    "delta_fitness": 0.005, # Min fitness improvement to be considered significant
    "N_stable": 3,          # Number of stable generations (no significant improvement) before stopping
    "D_min": 0.1,           # Minimum population diversity (not actively used in stopping criteria in current optimizer.py)
    "w_exec": 0.6,          # Weight for the execution score component of fitness
    "w_log": 0.4,           # Weight for the log/auxiliary score component of fitness
    "epsilon": 0.15,        # Weight for the replay buffer's contribution to fitness
    "alpha": 0.6,           # Smoothing factor for Exponential Moving Average (EMA) of fitness
    "seed_prompts": [       # Initial prompts to seed the first generation
        "Provide an accurate and thorough answer to the following question: ",
        "Answer with accuracy and thoroughness: ",
        "Explain the following topic clearly and in detail: "
    ]
}

# You can add other configurations here as your project grows.
# For example:
# LLM_SETTINGS = {
#     "model_name": "your-llm-model-identifier",
#     "api_key_env_var": "YOUR_LLM_API_KEY",
#     "request_timeout": 60 # seconds
# }
#
# LOGGING_CONFIG = {
#     "level": "INFO", # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
#     "format": "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
# }
