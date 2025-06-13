import json
import matplotlib.pyplot as plt
import os

def plot_evolution_history(file_path: str, output_dir: str = "./plots"):
    """
    Loads evolution history from a JSON file and plots best smoothed fitness and population diversity.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: Evolution history file not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return

    if not history:
        print("No evolution history data to plot.")
        return

    generations = [entry['generation'] for entry in history]
    best_fitness = [entry['best_smoothed_fitness'] for entry in history]
    diversity = [entry['population_diversity'] for entry in history]

    # Create plots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Smoothed Fitness', color=color)
    ax1.plot(generations, best_fitness, color=color, marker='o', linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Prompt Optimizer Evolution Over Generations')
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Population Diversity', color=color)  # we already handled the x-label with ax1
    ax2.plot(generations, diversity, color=color, marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "evolution_plot.png")
    plt.savefig(plot_path)
    print(f"Evolution plot saved to {plot_path}")
    plt.close(fig) # Close the plot to free memory

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    history_file = os.path.join(project_root, "result", "history.json")
    plot_output_dir = os.path.join(project_root, "plots")
    plot_evolution_history(history_file, plot_output_dir)

