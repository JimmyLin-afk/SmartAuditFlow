"""
AI Models configuration for the Smart Contract Audit application
"""
import os
from dotenv import load_dotenv

load_dotenv()

# AI Model configurations
AI_MODELS = {
    'gemini': {
        'api_key': os.getenv('GEMINI_API_KEY'),
        'model_name': 'gemini-2.0-flash-exp',
        'max_tokens': 8000,
        'temperature': 0.7,
        'enabled': bool(os.getenv('GEMINI_API_KEY'))
    },
    'openai': {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model_name': 'gpt-4o',
        'max_tokens': 8000,
        'temperature': 0.7,
        'enabled': bool(os.getenv('OPENAI_API_KEY'))
    },
    'claude': {
        'api_key': os.getenv('CLAUDE_API_KEY'),
        'model_name': 'claude-3-7-sonnet-latest',
        'max_tokens': 4000,
        'temperature': 0.7,
        'enabled': bool(os.getenv('CLAUDE_API_KEY'))
    }
}

# Default model priority (first available will be used)
MODEL_PRIORITY = ['gemini', 'openai', 'claude']

# Retry configuration
RETRY_CONFIG = {
    'max_retries': 3,
    'retry_enabled': True,
    'retry_interval': 1000  # milliseconds
}

# Model-specific prompts and templates
SYSTEM_PROMPTS = {
    'solidity_expert': "You are an expert in solidity programming language and smart contract analysis.",
    'security_auditor': "You are a security expert specializing in smart contract auditing and vulnerability detection.",
    'defi_specialist': "You are a DeFi specialist with expertise in decentralized finance protocols and their security implications."
}

# Knowledge base configuration for RAG
KNOWLEDGE_BASE_CONFIG = {
    'enabled': True,
    'embedding_model': 'text-embedding-ada-002',  # OpenAI embedding model
    'similarity_threshold': 0.7,
    'max_results': 5
}

