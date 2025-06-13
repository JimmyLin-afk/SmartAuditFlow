import json
import os
import sys

from apps.optimizer import IterativePromptOptimizer
from apps.llm_interface import M_llm_output
from apps.config import OPTIMIZER_CONFIG

def load_data_from_json(file_path: str) -> list:
    """Loads data (list of [context, answer] pairs) from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list) or not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in data):
                print(f"Warning: Data in {file_path} is not in the expected format (list of [context, answer]).")
                return []
            
            # Convert the 'answer' part of each item to a JSON string if it's a dictionary
            processed_data = []
            for context, answer in data:
                if isinstance(answer, dict):
                    processed_data.append((context, json.dumps(answer)))
                else:
                    processed_data.append((context, answer))
            return processed_data
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []

def save_output_to_file(output_data, directory: str, filename: str):
    """Saves the given data to a file in the specified directory."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return

    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(output_data, list):
                for item in output_data:
                    f.write(str(item) + '\n')
            else:
                f.write(str(output_data))
        print(f"Successfully saved output to {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")


if __name__ == '__main__':
    print("Starting Prompt Optimization Process...")

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    train_data_path = os.path.join(data_dir, "train_data.json")
    val_data_path = os.path.join(data_dir, "val_data.json")
    result_dir = os.path.join(project_root, "result")

    print(f"Loading training data from: {train_data_path}")
    d_train_data = load_data_from_json(train_data_path)
    print(f"Loading validation data from: {val_data_path}")
    d_val_data = load_data_from_json(val_data_path)

    if not d_train_data:
        print("Critical Error: Training data is empty or could not be loaded. Exiting.")
        sys.exit(1)

    print("Initializing the IterativePromptOptimizer...")
    try:
        prompt_optimizer = IterativePromptOptimizer(
            D_train=d_train_data,
            D_val=d_val_data,
            **OPTIMIZER_CONFIG,
            history_file_path = os.path.join(result_dir, "evaluation_history.json")
        )
    except Exception as e:
        print(f"Error during optimizer initialization: {e}")
        sys.exit(1)

    print("Starting optimization...")
    optimal_instruction = prompt_optimizer.optimize()

    print("\n--- Optimization Process Finished ---")
    if optimal_instruction and not (isinstance(optimal_instruction, str) and optimal_instruction.startswith("Error:")):
        print(f"The final optimal instruction found is: \'{optimal_instruction}\'")

        save_output_to_file(optimal_instruction, result_dir, "output.txt")



        print("\n--- Testing Optimal Instruction (Example) ---")
        test_context = "What is the primary benefit of evolutionary algorithms for complex problems?"       
        prompt_to_test = ""
        if isinstance(optimal_instruction, list):
            if optimal_instruction:
                prompt_to_test = str(optimal_instruction[0])
                print(f"Using the first item from the list of optimal instructions for testing: '{prompt_to_test}'")
            else:
                print("Optimal instruction is an empty list, cannot test.")
        else:
            prompt_to_test = str(optimal_instruction)

        if prompt_to_test:
            print(f"Using optimal prompt: '{prompt_to_test}'")
            print(f"On test context: '{test_context}'")
            llm_response = M_llm_output(prompt_to_test, test_context)
            print(f"LLM Response:")
            print(llm_response)
    else:
        print(f"Optimization did not successfully return a prompt or an error occurred: {optimal_instruction}")
        save_output_to_file(optimal_instruction, result_dir, "output.txt")


