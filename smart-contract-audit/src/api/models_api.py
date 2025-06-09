from flask import Blueprint, jsonify
from src.workflow.models import ModelManager

# Create blueprint for model endpoints
models_bp = Blueprint('models', __name__)

# Global model manager
global_model_manager = None

def get_model_manager():
    """Get or create global model manager"""
    global global_model_manager
    if global_model_manager is None:
        global_model_manager = ModelManager()
        print("🤖 Model manager initialized for API")
    return global_model_manager

@models_bp.route('/status', methods=['GET'])
def get_model_status():
    """Get availability status of all AI models"""
    try:
        model_manager = get_model_manager()
        status = model_manager.get_available_models()
        
        return jsonify({
            'status': 'success',
            'models': status,
            'gemini': status.get('gemini', False),
            'openai': status.get('openai', False),
            'claude': status.get('claude', False)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'models': {
                'gemini': False,
                'openai': False,
                'claude': False
            }
        }), 500

@models_bp.route('/test/<model_name>', methods=['POST'])
def test_model(model_name):
    """Test connection to a specific model"""
    try:
        model_manager = get_model_manager()
        
        if model_name not in ['gemini', 'openai', 'deepseek', 'claude']:
            return jsonify({
                'status': 'error',
                'error': f'Unknown model: {model_name}'
            }), 400
        
        # Test the specific model
        test_prompt = "Hello, this is a test message. Please respond with 'Test successful'."
        
        try:
            result = model_manager.call_model_sync(test_prompt, preferred_model=model_name)
            
            return jsonify({
                'status': 'success',
                'model': model_name,
                'available': True,
                'test_result': result[:100] + '...' if len(result) > 100 else result,
                'message': f'{model_name.title()} model is working correctly'
            }), 200
            
        except Exception as model_error:
            return jsonify({
                'status': 'error',
                'model': model_name,
                'available': False,
                'error': str(model_error),
                'message': f'{model_name.title()} model is not available'
            }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@models_bp.route('/info', methods=['GET'])
def get_model_info():
    """Get detailed information about all available models"""
    try:
        model_manager = get_model_manager()
        status = model_manager.get_available_models()
        
        model_info = {
            'gemini': {
                'name': 'Google Gemini',
                'description': 'Google\'s latest AI model with excellent code analysis capabilities',
                'speed': 'Fast',
                'accuracy': 'High',
                'available': status.get('gemini', False),
                'recommended_for': ['Smart contract analysis', 'Security auditing', 'Code review']
            },
            'openai': {
                'name': 'OpenAI GPT-4o',
                'description': 'Advanced reasoning model with proven track record in security analysis',
                'speed': 'Medium',
                'accuracy': 'Very High',
                'available': status.get('openai', False),
                'recommended_for': ['Complex security analysis', 'Detailed explanations', 'Risk assessment']
            },
            'deepseek': {
                'name': 'DeepSeek Chat',
                'description': 'Advanced reasoning model optimized for code understanding and analysis',
                'speed': 'Fast',
                'accuracy': 'High',
                'available': status.get('deepseek', False),
                'recommended_for': ['Code analysis', 'Pattern recognition', 'Technical documentation']
            },
            'claude': {
                'name': 'Anthropic Claude',
                'description': 'Helpful AI assistant with strong analytical and reasoning capabilities',
                'speed': 'Medium',
                'accuracy': 'Very High',
                'available': status.get('claude', False),
                'recommended_for': ['Code review', 'Security analysis', 'Best practices']
            },
            'auto': {
                'name': 'Auto Selection',
                'description': 'Automatically selects the best available model with intelligent fallback',
                'speed': 'Variable',
                'accuracy': 'High',
                'available': any(status.values()),
                'recommended_for': ['General use', 'Reliable fallback', 'Best performance']
            }
        }
        
        return jsonify({
            'status': 'success',
            'models': model_info,
            'availability_summary': status
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

