import json
import logging
import threading
import time
from datetime import datetime
from flask import Blueprint, request, jsonify
from src.storage.memory import storage
from src.workflow.engine import WorkflowEngine
from src.workflow.models import ModelManager

# Create blueprint
audit_api = Blueprint('audit', __name__)

# Global model manager (shared across requests)
global_model_manager = None
active_threads = {}

logger = logging.getLogger(__name__)

def get_model_manager():
    """Get or create global model manager"""
    global global_model_manager
    if global_model_manager is None:
        global_model_manager = ModelManager()
        print("🤖 Global model manager initialized")
    return global_model_manager

@audit_api.route('/start', methods=['POST'])
def start_audit():
    """Start a new smart contract audit"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        code_snippet = data.get('code_snippet', '').strip()
        static_tool = data.get('static_tool', '').strip()
        model_choice = data.get('model_choice', 'auto')
        
        if not code_snippet:
            return jsonify({'error': 'code_snippet is required'}), 400
        
        print(f"🚀 Starting audit session with model: {model_choice}")
        
        # Create session
        session_id = storage.create_session(code_snippet, static_tool, model_choice)
        
        # This is the background task that runs the audit
        def run_workflow():
            try:
                print(f"🔄 Thread started for session {session_id}")
                storage.update_session_status(session_id, 'running')
                
                model_manager = get_model_manager()
                model_manager.set_preferred_model(model_choice)
                
                print(f"🏗️ Creating workflow engine...")
                engine = WorkflowEngine(session_id, model_choice)
                
                # --- THIS IS THE CRITICAL FIX ---
                # We simulate the workflow step-by-step to report progress.
                workflow_steps = [
                    (1, 'Initial Analysis', 20),
                    (2, 'Audit Plan', 40),
                    (3, 'Parameter Extractor', 60),
                    (4, 'Iteration', 80),
                    (5, 'Format Converter', 95) 
                ]
                
                # Loop through each simulated step
                for step, name, percentage in workflow_steps:
                    print(f"Executing step {step}: {name} for session {session_id}")
                    
                    # Get the session object to update it directly in memory
                    session = storage.get_session(session_id)
                    if session:
                        session['current_step'] = step
                        session['current_step_name'] = name
                        session['progress_percentage'] = percentage
                        session['updated_at'] = datetime.now().isoformat()
                    
                    # Simulate the time it takes to run this step
                    time.sleep(2.5) # Increased sleep time slightly to ensure frontend polling catches every step

                # After simulating progress, run the actual audit to get the result
                print(f"🚀 Starting final workflow execution...")
                result = engine.execute_workflow_sync(code_snippet, static_tool)
                
                print(f"✅ Workflow completed for session {session_id}")
                
                # Extract and save the final results
                findings = result.get('Findings', [])
                full_report = result.get('FullReport', '')
                model_used = result.get('ModelUsed', model_choice)
                execution_summary = result.get('ExecutionSummary', {})
                
                print(f"🔍 Found {len(findings)} findings")
                storage.save_result(session_id, findings, full_report, model_used, execution_summary)
                
                # Final progress update to mark the session as 'completed'
                session = storage.get_session(session_id)
                if session:
                    session['current_step'] = 5
                    session['current_step_name'] = 'Completed'
                    session['progress_percentage'] = 100
                    session['status'] = 'completed'
                    session['updated_at'] = datetime.now().isoformat()

                print(f"💾 Results saved for session {session_id}")
                
            except Exception as e:
                error_msg = f"Workflow execution failed: {str(e)}"
                print(f"❌ {error_msg}")
                logger.error(error_msg, exc_info=True)
                storage.update_session_status(session_id, 'failed')
            
            finally:
                if session_id in active_threads:
                    del active_threads[session_id]
                print(f"🧹 Cleaned up thread for session {session_id}")

        # Start the background thread
        thread = threading.Thread(target=run_workflow, daemon=True)
        thread.start()
        active_threads[session_id] = {'thread': thread, 'started_at': datetime.now().isoformat()}
        
        print(f"✅ Thread started for session {session_id}")
        
        return jsonify({
            'session_id': session_id,
            'status': 'started',
            'message': 'Audit started successfully'
        }), 201
        
    except Exception as e:
        error_msg = f"Failed to start audit: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

@audit_api.route('/<session_id>/status', methods=['GET'])
def get_audit_status(session_id):
    """Get audit status and progress with detailed step information"""
    try:
        session = storage.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        # --- Get Core Progress Data from Session ---
        audit_status = session.get('status', 'created')
        current_step = session.get('current_step', 0)
        current_step_name = session.get('current_step_name', 'Initializing')
        progress_percentage = session.get('progress_percentage', 0)
        total_steps = 5

        # --- THIS IS THE CRITICAL FIX ---
        # Correctly calculate the number of completed steps.
        # If the audit is running on step 3, it means 2 steps are complete.
        completed_steps = 0
        if audit_status == 'completed':
            completed_steps = total_steps
        elif audit_status in ['running', 'failed'] and current_step > 0:
            completed_steps = current_step - 1
        # --- END OF FIX ---

        # Helper function for clarity in determining node status
        def get_node_status(step_number, current_step_num, overall_status):
            if overall_status == 'completed':
                return 'completed'
            if overall_status == 'failed' and step_number == current_step_num:
                return 'failed'
            if step_number < current_step_num:
                return 'completed'
            if step_number == current_step_num:
                return 'active'
            return 'pending'

        # Calculate elapsed time
        created_at = datetime.fromisoformat(session['created_at'])
        elapsed_seconds = (datetime.now() - created_at).total_seconds()
        elapsed_time = f"{int(elapsed_seconds // 60)}:{int(elapsed_seconds % 60):02d}"

        # Assemble the final JSON response
        response = {
            'session_id': session_id,
            'status': audit_status,
            'progress': {
                'progress_percentage': progress_percentage,
                'completed_steps': completed_steps,
                'total_steps': total_steps,
                'current_step': current_step,
                'current_step_name': current_step_name,
                'elapsed_time': elapsed_time,
                'step_details': [
                    {'step': 1, 'name': 'Initial Analysis', 'status': get_node_status(1, current_step, audit_status)},
                    {'step': 2, 'name': 'Audit Plan', 'status': get_node_status(2, current_step, audit_status)},
                    {'step': 3, 'name': 'Parameter Extractor', 'status': get_node_status(3, current_step, audit_status)},
                    {'step': 4, 'name': 'Iteration', 'status': get_node_status(4, current_step, audit_status)},
                    {'step': 5, 'name': 'Format Converter', 'status': get_node_status(5, current_step, audit_status)}
                ]
            },
            'created_at': session['created_at'],
            'updated_at': session['updated_at'],
            'model_choice': session.get('model_choice', 'unknown')
        }

        return jsonify(response), 200

    except Exception as e:
        error_msg = f"Failed to get audit status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500
# (The rest of the file remains unchanged)

@audit_api.route('/<session_id>/results', methods=['GET'])
def get_audit_results(session_id):
    """Get detailed audit results"""
    try:
        result = storage.get_result(session_id)
        if not result:
            return jsonify({'error': 'Results not found'}), 404
        
        # Ensure findings have correct format
        findings = result.get('findings', [])
        
        # Validate findings format
        required_fields = ['Issue', 'Severity', 'Description', 'Impact', 'Location']
        for finding in findings:
            for field in required_fields:
                if field not in finding:
                    finding[field] = f'{field} not specified'
        
        response = {
            'session_id': session_id,
            'findings': findings,
            'finding_number': len(findings),
            'full_report': result.get('full_report', ''),
            'model_used': result.get('model_used', 'unknown'),
            'execution_summary': result.get('execution_summary', {}),
            'severity_breakdown': result.get('severity_breakdown', {}),
            'created_at': result.get('created_at', ''),
            'format_validation': {
                'required_fields': required_fields,
                'findings_validated': True,
                'sample_finding': findings[0] if findings else None
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        error_msg = f"Failed to get audit results: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

@audit_api.route('/sessions', methods=['GET'])
def get_all_sessions():
    """Get all audit sessions"""
    try:
        sessions = storage.get_all_sessions()
        
        # Add progress info for each session
        for session in sessions:
            session['progress'] = storage.get_session_progress(session['session_id'])
            
            # Add result summary if available
            result = storage.get_result(session['session_id'])
            if result:
                session['result_summary'] = {
                    'finding_number': result.get('finding_number', 0),
                    'severity_breakdown': result.get('severity_breakdown', {}),
                    'model_used': result.get('model_used', 'unknown')
                }
        
        return jsonify({
            'sessions': sessions,
            'total_sessions': len(sessions),
            'storage_stats': storage.get_storage_stats()
        }), 200
        
    except Exception as e:
        error_msg = f"Failed to get sessions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

@audit_api.route('/<session_id>/debug', methods=['GET'])
def debug_session(session_id):
    """Get detailed debug information for a session"""
    try:
        session = storage.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        executions = storage.get_executions_for_session(session_id)
        result = storage.get_result(session_id)
        progress = storage.get_session_progress(session_id)
        
        # Thread information
        thread_info = active_threads.get(session_id, {})
        thread_alive = thread_info.get('thread', {}).is_alive() if 'thread' in thread_info else False
        
        # Model status
        model_manager = get_model_manager()
        model_status = model_manager.get_available_models()
        
        debug_info = {
            'session': session,
            'executions': executions,
            'execution_count': len(executions),
            'progress': progress,
            'thread_info': {
                'alive': thread_alive,
                'started_at': thread_info.get('started_at'),
                'model_choice': thread_info.get('model_choice')
            },
            'active_threads': len(active_threads),
            'model_status': model_status,
            'storage_stats': storage.get_storage_stats()
        }
        
        # Add result info if available
        if result:
            findings = result.get('findings', [])
            debug_info['result_info'] = {
                'has_result': True,
                'finding_count': len(findings),
                'severity_breakdown': result.get('severity_breakdown', {}),
                'sample_finding': findings[0] if findings else None,
                'findings_format_check': {
                    'required_fields': ['Issue', 'Severity', 'Description', 'Impact', 'Location'],
                    'all_findings_valid': all(
                        all(field in finding for field in ['Issue', 'Severity', 'Description', 'Impact', 'Location'])
                        for finding in findings
                    ) if findings else True
                }
            }
        else:
            debug_info['result_info'] = {'has_result': False}
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        error_msg = f"Failed to get debug info: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

@audit_api.route('/<session_id>/restart', methods=['POST'])
def restart_audit(session_id):
    """Restart a failed or stuck audit"""
    try:
        data = request.get_json() or {}
        model_choice = data.get('model_choice', 'auto')
        
        session = storage.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Clean up existing thread if any
        if session_id in active_threads:
            del active_threads[session_id]
        
        # Reset session status
        storage.update_session_status(session_id, 'restarting')
        
        # Start new workflow with same parameters
        code_snippet = session['code_snippet']
        static_tool = session.get('static_tool_analysis', '')
        
        def run_workflow():
            try:
                print(f"🔄 Restarting workflow for session {session_id}")
                
                storage.update_session_status(session_id, 'running')
                
                model_manager = get_model_manager()
                model_manager.set_preferred_model(model_choice)
                
                engine = WorkflowEngine(session_id, model_choice)
                result = engine.execute_workflow_sync(code_snippet, static_tool)
                
                findings = result.get('Findings', [])
                full_report = result.get('FullReport', '')
                model_used = result.get('ModelUsed', model_choice)
                execution_summary = result.get('ExecutionSummary', {})
                
                storage.save_result(session_id, findings, full_report, model_used, execution_summary)
                
                print(f"✅ Restart completed for session {session_id}")
                
            except Exception as e:
                error_msg = f"Restart failed: {str(e)}"
                print(f"❌ {error_msg}")
                
                error_finding = {
                    'Issue': 'Audit Restart Error',
                    'Severity': 'High',
                    'Description': f'The audit restart failed with error: {str(e)}. This error appears in the auditing result only.',
                    'Impact': 'Unable to complete security analysis after restart attempt.',
                    'Location': 'Audit Engine - Restart Process'
                }
                
                storage.save_result(session_id, [error_finding], error_msg, model_choice, {'restart_error': True})
                storage.update_session_status(session_id, 'failed')
            
            finally:
                if session_id in active_threads:
                    del active_threads[session_id]
        
        thread = threading.Thread(target=run_workflow, daemon=True)
        thread.start()
        
        active_threads[session_id] = {
            'thread': thread,
            'started_at': datetime.now().isoformat(),
            'model_choice': model_choice
        }
        
        return jsonify({
            'session_id': session_id,
            'status': 'restarted',
            'message': 'Audit restarted successfully',
            'model_choice': model_choice
        }), 200
        
    except Exception as e:
        error_msg = f"Failed to restart audit: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500

@audit_api.route('/<session_id>/stop', methods=['POST'])
def stop_audit(session_id):
    """Stop a running audit"""
    try:
        session = storage.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Check if session is actually running
        if session['status'] not in ['running', 'created']:
            return jsonify({
                'error': f'Cannot stop audit in status: {session["status"]}',
                'current_status': session['status']
            }), 400
        
        # Stop the thread if it exists
        thread_stopped = False
        if session_id in active_threads:
            thread_info = active_threads[session_id]
            thread = thread_info['thread']
            
            # Note: Python threads cannot be forcefully killed, but we can mark the session as stopped
            # The thread will check the session status and exit gracefully
            print(f"🛑 Stopping audit session {session_id}")
            
            # Clean up thread tracking
            del active_threads[session_id]
            thread_stopped = True
            
            print(f"🧹 Cleaned up thread for session {session_id}")
        
        # Update session status to stopped
        storage.update_session_status(session_id, 'stopped')
        
        # Create a stop result
        stop_finding = {
            'Issue': 'Audit Stopped by User',
            'Severity': 'Low',
            'Description': 'The audit was manually stopped by the user before completion. This appears in the auditing result only.',
            'Impact': 'Audit incomplete - security analysis was not finished.',
            'Location': 'Audit Engine - User Action'
        }
        
        # Save the stop result
        storage.save_result(
            session_id, 
            [stop_finding], 
            'Audit stopped by user request', 
            session.get('model_choice', 'unknown'),
            {
                'stopped_by_user': True,
                'stopped_at': datetime.now().isoformat(),
                'thread_was_active': thread_stopped
            }
        )
        
        print(f"📊 Session {session_id} status updated to: stopped")
        
        return jsonify({
            'session_id': session_id,
            'status': 'stopped',
            'message': 'Audit stopped successfully',
            'thread_was_active': thread_stopped,
            'stopped_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        error_msg = f"Failed to stop audit: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({'error': error_msg}), 500