import random
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained Sentence-BERT model
# This will download the model the first time it's run.
# Consider moving this to a more central place or initializing once if performance is critical.
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_prompt_similarity(prompt1: str, prompt2: str) -> float:
    """
    Calculates similarity between two prompts using Sentence-BERT embeddings and cosine similarity.
    Returns a score between 0 and 1.
    """
    if not prompt1 and not prompt2:
        return 1.0
    if not prompt1 or not prompt2:
        return 0.0

    # Encode the prompts to get their embeddings
    embedding1 = model.encode(prompt1, convert_to_tensor=True)
    embedding2 = model.encode(prompt2, convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_similarity = util.cos_sim(embedding1, embedding2).item()

    # Cosine similarity ranges from -1 to 1. Normalize to 0 to 1.
    return (cosine_similarity + 1) / 2

def mutate_prompt_guided(parent_prompt: str, temperature: float, current_batch: list) -> str:
    """
    Mutates a parent prompt using paraphrasing, keyword substitution, and structural edits.
    `temperature` controls the extent/randomness of mutation.
    `current_batch` (list of (context, answer) tuples) can be used for guided mutation.
    """
    words = parent_prompt.split()
    if not words:
        return "Analyze the smart contract thoroughly."

    # Determine number of mutations based on temperature and prompt length
    num_mutations = max(1, int(len(words) * temperature * 0.3)) # Reduced impact of temperature

    for _ in range(num_mutations):
        if not words: break
        mutation_type = random.choice([
            "replace_keyword", 
            "insert_phrase", 
            "delete_word", 
            "reorder_words",
            "paraphrase_segment"
        ])

        if mutation_type == "replace_keyword" and words:
            # More relevant keywords for smart contract analysis
            keywords_to_replace = {
                "analyze": ["examine", "inspect", "review", "evaluate"],
                "focus": ["concentrate on", "highlight", "emphasize"],
                "list": ["enumerate", "detail", "provide", "identify"],
                "precise": ["accurate", "exact", "specific"],
                "comprehensive": ["thorough", "complete", "exhaustive"],
                "detail": ["describe", "explain", "outline"],
                "purpose": ["functionality", "role", "objective"],
                "conditions": ["requirements", "criteria", "checks"],
                "emitted": ["triggered", "logged", "dispatched"]
            }
            
            # Find a word to replace
            eligible_words = [w for w in words if w.lower() in keywords_to_replace]
            if eligible_words:
                word_to_replace = random.choice(eligible_words)
                new_word = random.choice(keywords_to_replace[word_to_replace.lower()])
                idx = words.index(word_to_replace)
                words[idx] = new_word # Keep original casing for now, or apply new_word.capitalize() etc.

        elif mutation_type == "insert_phrase":
            phrases_to_insert = [
                "Ensure clarity.", 
                "Provide a summary.", 
                "Include all relevant details.",
                "Pay attention to visibility.",
                "Consider all parameters."
            ]
            idx = random.randint(0, len(words))
            words.insert(idx, random.choice(phrases_to_insert))

        elif mutation_type == "delete_word" and len(words) > 3: # Avoid making prompts too short
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)

        elif mutation_type == "reorder_words" and len(words) > 2:
            # Reorder a small segment of words
            start_idx = random.randint(0, len(words) - 2)
            end_idx = random.randint(start_idx + 1, len(words) - 1)
            segment = words[start_idx:end_idx+1]
            random.shuffle(segment)
            words[start_idx:end_idx+1] = segment

        elif mutation_type == "paraphrase_segment" and len(words) > 5:
            # This is a very basic form of paraphrasing by replacing a common phrase
            # In a real scenario, this would involve a more sophisticated NLP technique
            replacements = {
                "Focus on its function definitions": "Highlight functions",
                "List all functions": "Enumerate functions",
                "Be precise": "Ensure accuracy"
            }
            original_prompt_str = " ".join(words)
            for old_phrase, new_phrase in replacements.items():
                if old_phrase in original_prompt_str:
                    original_prompt_str = original_prompt_str.replace(old_phrase, new_phrase, 1)
                    words = original_prompt_str.split()
                    break

    # Guided mutation: add a word from a context in the batch if relevant
    if current_batch and random.random() < 0.2: # 20% chance
        sample_context_item = random.choice(current_batch)
        sample_context_text = sample_context_item[0] # Assuming (context, answer) structure
        
        # Extract potential keywords from the smart contract context
        # This is a very basic extraction, could be improved with AST parsing etc.
        contract_keywords = []
        if "function" in sample_context_text.lower(): contract_keywords.append("function")
        if "event" in sample_context_text.lower(): contract_keywords.append("event")
        if "modifier" in sample_context_text.lower(): contract_keywords.append("modifier")
        if "mapping" in sample_context_text.lower(): contract_keywords.append("mapping")
        if "address" in sample_context_text.lower(): contract_keywords.append("address")
        if "uint" in sample_context_text.lower(): contract_keywords.append("uint")

        if contract_keywords:
            chosen_word = random.choice(contract_keywords)
            if chosen_word not in [w.lower() for w in words]: # Avoid adding duplicates
                 words.append(chosen_word)

    return " ".join(words) if words else "Analyze the smart contract."


