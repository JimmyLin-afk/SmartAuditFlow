import random
from Levenshtein import distance as levenshtein_distance # pip install python-Levenshtein

def calculate_prompt_similarity(prompt1: str, prompt2: str) -> float:
    """
    Calculates similarity between two prompts. Returns a score between 0 and 1.
    (1 for identical, 0 for completely different).

    Args:
        prompt1 (str): The first prompt.
        prompt2 (str): The second prompt.

    Returns:
        float: The similarity score.
    """
    # Placeholder: Using normalized Levenshtein distance as an example.
    # Consider semantic similarity for more advanced use cases.
    if not prompt1 and not prompt2:
        return 1.0
    if not prompt1 or not prompt2: # One is empty, the other is not
        return 0.0
    max_len = max(len(prompt1), len(prompt2))
    if max_len == 0: # Should be caught by the first condition
        return 1.0
    return 1.0 - (levenshtein_distance(prompt1, prompt2) / max_len)

def mutate_prompt_guided(parent_prompt: str, temperature: float, current_batch: list) -> str:
    """
    Mutates a parent prompt.
    'temperature' can control the extent/randomness of mutation.
    'current_batch' (list of (context, answer) tuples) can be used for guided mutation.

    Args:
        parent_prompt (str): The prompt to mutate.
        temperature (float): Controls the mutation strength/randomness.
        current_batch (list): Mini-batch of data for context-aware mutation.

    Returns:
        str: The mutated prompt.
    """
    # Placeholder: This is a very basic mutation strategy.
    # You'll likely want to implement more sophisticated prompt mutation techniques.
    words = parent_prompt.split()
    if not words:
        return "A new simple example prompt." # Default for empty parent

    num_mutations = max(1, int(len(words) * temperature * 0.5)) # Higher temp = more potential changes

    for _ in range(num_mutations):
        if not words: break # Stop if words list becomes empty
        mutation_type = random.choice(["replace_word", "insert_word", "delete_word", "swap_words"])

        if mutation_type == "replace_word" and words:
            idx = random.randint(0, len(words) - 1)
            words[idx] = random.choice(["optimal", "concise", "detailed", "effective", "new", "sample", "response", "instruction"])
        elif mutation_type == "insert_word":
            idx = random.randint(0, len(words)) # Can insert at the end
            words.insert(idx, random.choice(["please", "ensure", "generate", "provide", "create", "summarize", "explain"]))
        elif mutation_type == "delete_word" and len(words) > 1: # Avoid deleting the last word
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        elif mutation_type == "swap_words" and len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

    # Example of "guided" mutation (very naive): add a word from a context in the batch
    if current_batch and random.random() < 0.3: # 30% chance to add a word from context
        sample_context_item = random.choice(current_batch)
        sample_context_text = sample_context_item[0] # Assuming (context, answer) structure
        context_words = sample_context_text.split()
        if context_words:
            chosen_word = random.choice(context_words)
            # Add word if it's not too long and not already present (simple checks)
            if len(chosen_word) < 15 and chosen_word not in words:
                 words.append(chosen_word.strip(".,?!"))


    return " ".join(words) if words else "Mutated into an empty prompt."
