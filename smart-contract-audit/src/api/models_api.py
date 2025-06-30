from flask import Blueprint, jsonify
from src.workflow.models import get_model_manager

# Create blueprint for model endpoints
models_bp = Blueprint('models', __name__)

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
            'gpt-4o-mini': status.get('gpt-4o-mini', False),
            'gpt-4.1': status.get('gpt-4.1', False),
            'o4': status.get('o4', False),
            'o3-mini': status.get('o3-mini', False),
            'deepseek': status.get('deepseek', False),
            'claude': status.get('claude', False),
            'claude-opus-4-0': status.get('claude-opus-4-0', False),
            'claude-sonnet-4-0': status.get('claude-sonnet-4-0', False),
            'claude-3-7-sonnet-latest': status.get('claude-3-7-sonnet-latest', False)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'models': {
                'gemini': False,
                'openai': False,
                'gpt-4o-mini': False,
                'gpt-4.1': False,
                'o4': False,
                'o3-mini': False,
                'deepseek': False,
                'claude': False,
                'claude-opus-4-0': False,
                'claude-sonnet-4-0': False,
                'claude-3-7-sonnet-latest': False
            }
        }), 500

@models_bp.route('/test/<model_name>', methods=['POST'])
def test_model(model_name):
    """Test connection to a specific model"""
    try:
        model_manager = get_model_manager()
        
        if model_name not in ['gemini', 'openai', 'gpt-4o-mini', 'gpt-4.1', 'o4', 'o3-mini', 'deepseek', 'claude', 'claude-opus-4-0', 'claude-sonnet-4-0', 'claude-3-7-sonnet-latest']:
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
            'gpt-4o-mini': {
                'name': 'OpenAI GPT-4o-mini',
                'description': 'Efficient and cost-effective version of GPT-4o with excellent performance for code analysis',
                'speed': 'Fast',
                'accuracy': 'High',
                'available': status.get('gpt-4o-mini', False),
                'recommended_for': ['Cost-effective analysis', 'Quick security checks', 'Batch processing']
            },
            'gpt-4.1': {
                'name': 'OpenAI GPT-4.1',
                'description': 'Enhanced version of GPT-4 with improved reasoning and code analysis capabilities',
                'speed': 'Medium',
                'accuracy': 'Very High',
                'available': status.get('gpt-4.1', False),
                'recommended_for': ['Advanced security analysis', 'Complex vulnerability detection', 'Detailed code review']
            },
            'o4': {
                'name': 'OpenAI O4',
                'description': 'Advanced reasoning model with exceptional problem-solving capabilities for complex analysis',
                'speed': 'Medium',
                'accuracy': 'Very High',
                'available': status.get('o4', False),
                'recommended_for': ['Complex reasoning tasks', 'Advanced security analysis', 'Critical vulnerability assessment']
            },
            'o3-mini': {
                'name': 'OpenAI O3-mini',
                'description': 'Efficient reasoning model optimized for fast and accurate code analysis',
                'speed': 'Fast',
                'accuracy': 'High',
                'available': status.get('o3-mini', False),
                'recommended_for': ['Quick reasoning tasks', 'Efficient code analysis', 'Rapid security checks']
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
                'name': 'Anthropic Claude 3.5 Sonnet',
                'description': 'Helpful AI assistant with strong analytical and reasoning capabilities',
                'speed': 'Medium',
                'accuracy': 'Very High',
                'available': status.get('claude', False),
                'recommended_for': ['Code review', 'Security analysis', 'Best practices']
            },
            'claude-opus-4-0': {
                'name': 'Anthropic Claude Opus 4.0',
                'description': 'Most powerful Claude model with exceptional reasoning and analysis capabilities',
                'speed': 'Medium',
                'accuracy': 'Exceptional',
                'available': status.get('claude-opus-4-0', False),
                'recommended_for': ['Critical security analysis', 'Complex vulnerability assessment', 'Comprehensive code review']
            },
            'claude-sonnet-4-0': {
                'name': 'Anthropic Claude Sonnet 4.0',
                'description': 'Balanced Claude model optimized for performance and accuracy in security analysis',
                'speed': 'Medium',
                'accuracy': 'Very High',
                'available': status.get('claude-sonnet-4-0', False),
                'recommended_for': ['Balanced security analysis', 'Code quality assessment', 'Risk evaluation']
            },
            'claude-3-7-sonnet-latest': {
                'name': 'Anthropic Claude 3.7 Sonnet Latest',
                'description': 'Latest version of Claude Sonnet with enhanced capabilities for smart contract analysis',
                'speed': 'Medium',
                'accuracy': 'Very High',
                'available': status.get('claude-3-7-sonnet-latest', False),
                'recommended_for': ['Latest security patterns', 'Modern vulnerability detection', 'Advanced code analysis']
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
