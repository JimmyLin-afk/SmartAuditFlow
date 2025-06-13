import re
import json

def parse_smart_contract_analysis(analysis_text: str) -> dict:
    """
    Parses the smart contract analysis text (expected to be JSON string) to extract structured information.
    """
    parsed_data = {
        "functions": set(),
        "state_variables": set(),
        "modifiers": set(),
        "events": set(),
        "questions": set()
    }
    try:
        # Attempt to load as JSON
        data = json.loads(analysis_text)

        # Extract functions
        for func in data.get("functions", []):
            name = func.get("name")
            params = func.get("params")
            if name and params:
                parsed_data["functions"].add(f"{name}({params})".lower())
            elif name:
                parsed_data["functions"].add(name.lower())

        # Extract state variables
        for var in data.get("state_variables", []):
            name = var.get("name")
            var_type = var.get("type")
            if name and var_type:
                parsed_data["state_variables"].add(f"{var_type} {name}".lower())
            elif name:
                parsed_data["state_variables"].add(name.lower())

        # Extract modifiers
        for mod in data.get("modifiers", []):
            name = mod.get("name")
            if name:
                parsed_data["modifiers"].add(name.lower())

        # Extract events
        for event in data.get("events", []):
            name = event.get("name")
            params = event.get("params")
            if name and params:
                parsed_data["events"].add(f"{name}({params})".lower())
            elif name:
                parsed_data["events"].add(name.lower())

        # Extract questions
        for q in data.get("questions", []):
            if q:
                parsed_data["questions"].add(q.lower())

    except json.JSONDecodeError:
        # Fallback for non-JSON or malformed JSON output (e.g., from initial simple LLM sim)
        # This part can be removed once LLM always outputs structured JSON
        print("Warning: LLM output is not valid JSON. Attempting regex parsing.")
        # Functions
        func_matches = re.findall(r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\(.*?\)", analysis_text, re.IGNORECASE)
        for f in func_matches:
            parsed_data["functions"].add(f.lower())

        # State Variables
        var_matches = re.findall(r"(uint|int|address|bool|string|bytes)\s+(public|private|internal|external)?\s*([a-zA-Z_][a-zA-Z0-9_]*);", analysis_text, re.IGNORECASE)
        for _, _, var_name in var_matches:
            parsed_data["state_variables"].add(var_name.lower())

        # Modifiers
        mod_matches = re.findall(r"modifier\s+([a-zA-Z_][a-zA-Z0-9_]*)\(.*?\)", analysis_text, re.IGNORECASE)
        for m in mod_matches:
            parsed_data["modifiers"].add(m.lower())

        # Events
        event_matches = re.findall(r"event\s+([a-zA-Z_][a-zA-Z0-9_]*)\(.*?\);", analysis_text, re.IGNORECASE)
        for e in event_matches:
            parsed_data["events"].add(e.lower())

    return parsed_data

def calculate_f1_score(predicted_set: set, actual_set: set) -> float:
    """
    Calculates the F1 score for two sets.
    """
    if not actual_set and not predicted_set:
        return 1.0 # Perfect score if both are empty
    if not actual_set and predicted_set:
        return 0.0 # Predicted something when nothing was expected
    if actual_set and not predicted_set:
        return 0.0 # Missed everything

    intersection = len(predicted_set.intersection(actual_set))
    precision = intersection / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = intersection / len(actual_set) if len(actual_set) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def f_coverage(llm_output: str, target_answer: str, weights: dict) -> float:
    """
    Calculates the component coverage score based on F1 scores for each component type.
    weights: dict with keys 'w_F', 'w_V', 'w_M', 'w_E' for functions, variables, modifiers, events.
    """
    llm_parsed = parse_smart_contract_analysis(llm_output)
    target_parsed = parse_smart_contract_analysis(target_answer)

    f1_functions = calculate_f1_score(llm_parsed["functions"], target_parsed["functions"])
    f1_variables = calculate_f1_score(llm_parsed["state_variables"], target_parsed["state_variables"])
    f1_modifiers = calculate_f1_score(llm_parsed["modifiers"], target_parsed["modifiers"])
    f1_events = calculate_f1_score(llm_parsed["events"], target_parsed["events"])

    # Default weights if not provided, or from config
    w_F = weights.get("w_F", 0.25)
    w_V = weights.get("w_V", 0.25)
    w_M = weights.get("w_M", 0.25)
    w_E = weights.get("w_E", 0.25)

    total_weight = w_F + w_V + w_M + w_E
    if total_weight == 0:
        return 0.0

    score = (w_F * f1_functions + w_V * f1_variables + w_M * f1_modifiers + w_E * f1_events) / total_weight
    return score

def f_detail(llm_output: str, target_answer: str) -> float:
    """
    Simulates the detail accuracy score using a placeholder judge model.
    In a real scenario, this would involve another LLM evaluating the quality
    of descriptions.
    """
    score = 0.0
    # Parse both LLM output and target answer as JSON
    try:
        llm_data = json.loads(llm_output)
        target_data = json.loads(target_answer)
    except json.JSONDecodeError:
        # If parsing fails, fall back to simple string comparison or return 0
        return 0.0 # Or implement a different fallback

    # Heuristic for detail accuracy: compare presence of keys and length of descriptions
    # This is a very simplified version of what a judge model would do.
    
    # Check if all expected top-level keys are present in LLM output
    expected_keys = ["functions", "state_variables", "modifiers", "events", "questions"]
    for key in expected_keys:
        if key in target_data and key in llm_data and len(target_data[key]) > 0:
            # For each item in the target, check if a corresponding item exists in LLM output
            # and if its description has some length/content.
            # This is a very rough heuristic. A real judge model would compare descriptions semantically.
            llm_item_count = len(llm_data[key])
            target_item_count = len(target_data[key])
            
            if target_item_count > 0:
                # Award points for having items and for their descriptions being non-empty
                score_per_item = 0.2 / len(expected_keys) # Distribute score across component types
                if llm_item_count > 0:
                    score += score_per_item * min(llm_item_count / target_item_count, 1.0) # Score for quantity
                    
                    # Check for non-empty descriptions for a sample of items
                    sample_size = min(3, llm_item_count)
                    non_empty_desc_count = 0
                    for i in range(sample_size):
                        if key == "functions" and "purpose" in llm_data[key][i]:
                            if llm_data[key][i]["purpose"] and len(llm_data[key][i]["purpose"]) > 5:
                                non_empty_desc_count += 1
                        elif key == "state_variables" and "purpose" in llm_data[key][i]:
                            if llm_data[key][i]["purpose"] and len(llm_data[key][i]["purpose"]) > 5:
                                non_empty_desc_count += 1
                        elif key == "modifiers" and "conditions" in llm_data[key][i]:
                            if llm_data[key][i]["conditions"] and len(llm_data[key][i]["conditions"]) > 5:
                                non_empty_desc_count += 1
                        elif key == "events" and "emission_context" in llm_data[key][i]:
                            if llm_data[key][i]["emission_context"] and len(llm_data[key][i]["emission_context"]) > 5:
                                non_empty_desc_count += 1
                        elif key == "questions" and llm_data[key][i] and len(llm_data[key][i]) > 5:
                            non_empty_desc_count += 1

                    if sample_size > 0:
                        score += score_per_item * (non_empty_desc_count / sample_size) * 0.5 # Additional score for detail presence

    return min(score, 1.0)

def f_exec_score(llm_output: str, target_answer: str, w_cov: float, w_det: float, w_F: float, w_V: float, w_M: float, w_E: float) -> float:
    """
    Scores the execution/accuracy of the LLM output against the target answer
    for smart contract analysis, combining coverage and detail accuracy.
    """
    coverage_score = f_coverage(llm_output, target_answer, {"w_F": w_F, "w_V": w_V, "w_M": w_M, "w_E": w_E})
    detail_score = f_detail(llm_output, target_answer)
    
    return (w_cov * coverage_score) + (w_det * detail_score)

def f_log_score(llm_output: str, target_answer: str) -> float:
    """
    Simulates the log-likelihood score.
    In a real scenario, this would involve querying the LLM API for log-likelihoods.
    For now, a simple heuristic based on output quality.
    """
    # Placeholder: A simple heuristic for log-likelihood
    # Assume higher score for outputs that are well-formed and contain expected keywords
    score = 0.0
    try:
        llm_data = json.loads(llm_output)
        # Award points for valid JSON structure
        score += 0.2
        # Award points for having a reasonable number of top-level keys
        if len(llm_data.keys()) >= 4: # functions, state_variables, modifiers, events
            score += 0.2
    except json.JSONDecodeError:
        pass # No points for invalid JSON

    # Check for overall length (proxy for completeness/detail)
    if len(llm_output) > 200: # Longer, more detailed output might be preferred
        score += 0.3
    elif len(llm_output) > 50:
        score += 0.1

    # Check for presence of some descriptive keywords (very basic simulation of fluency/style)
    if "purpose" in llm_output.lower() or "conditions" in llm_output.lower() or "emission_context" in llm_output.lower():
        score += 0.3

    return min(score, 1.0)


