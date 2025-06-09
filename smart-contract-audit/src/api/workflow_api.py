from flask import Blueprint, request, jsonify

from src.storage.memory import storage
from src.workflow.engine import WorkflowEngine

workflow_api = Blueprint('workflow_api', __name__)

@workflow_api.route('/<session_id>/progress', methods=['GET'])
def get_workflow_progress(session_id):
    """Get workflow progress"""
    try:
        audit_session = storage.get_audit_session(session_id)
        
        if not audit_session:
            return jsonify({'error': 'Session not found'}), 404
        
        engine = WorkflowEngine(session_id)
        progress = engine.get_progress()
        
        return jsonify(progress), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@workflow_api.route('/<session_id>/logs', methods=['GET'])
def get_workflow_logs(session_id):
    """Get workflow execution logs"""
    try:
        audit_session = storage.get_audit_session(session_id)
        
        if not audit_session:
            return jsonify({'error': 'Session not found'}), 404
        
        engine = WorkflowEngine(session_id)
        logs = engine.get_logs()
        
        return jsonify({
            'session_id': session_id,
            'logs': logs
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

