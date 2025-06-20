import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def M_llm_output(prompt: str, context: str) -> str:
    """
    Calls the GPT-4o LLM to generate an output
    based on the given prompt and context for smart contract analysis.

    Args:
        prompt (str): The instruction for the LLM.
        context (str): The smart contract code or relevant context.

    Returns:
        str: The output generated by the LLM, containing extracted components
             in a structured (JSON string) format.
    """
    try:
        # Construct the messages for the chat completion API
        messages = [
            {"role": "system", "content": "You are an expert in smart contract analysis. Provide detailed analysis in JSON format."}, # System role for context
            {"role": "user", "content": f"Analyze the following smart contract code based on the instruction: {prompt}\n\nSmart Contract Code:\n{context}\n\nProvide the analysis in a JSON object with keys for \'functions\', \'state_variables\', \'modifiers\', \'events\', and \'questions\'. Each key should contain a list of objects with relevant details (e.g., for functions: name, params, visibility, returns, purpose; for state variables: name, type, visibility, purpose; for modifiers: name, conditions; for events: name, params, emission_context; for questions: list of strings). If a component is not found, its list should be empty."}
        ]

        # Make the API call to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={ "type": "json_object" }, # Request JSON output
            temperature=0.7, # Adjust temperature for creativity vs. determinism
            max_tokens=8000 # Limit response length to avoid excessive costs
        )

        # Extract the content from the response
        llm_response_content = response.choices[0].message.content
        
        # Validate and return JSON. If LLM doesn't return valid JSON, this will raise an error.
        # The `response_format` parameter should enforce JSON, but a fallback is good.
        try:
            json.loads(llm_response_content) # Just to validate it's JSON
            return llm_response_content
        except json.JSONDecodeError:
            print(f"Warning: LLM did not return valid JSON. Raw response: {llm_response_content}")
            # Fallback to a structured error message or attempt to fix/parse
            return json.dumps({"error": "LLM did not return valid JSON", "raw_response": llm_response_content})

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return json.dumps({"error": str(e), "message": "Failed to get response from LLM."})