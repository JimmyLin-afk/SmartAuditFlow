"""
In-memory storage for audit sessions and results
Updated to handle the new findings format
"""
import uuid
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class InMemoryStorage:
    """Simple in-memory storage for audit sessions and results"""
    
    def __init__(self):
        self.sessions = {}
        self.executions = {}
        self.results = {}
        self.execution_counter = 0
        
    def create_session(self, code_snippet: str, static_tool_analysis: str = None, model_choice: str = 'auto') -> str:
        """Create a new audit session"""
        session_id = str(uuid.uuid4())
        
        session = {
            'id': len(self.sessions) + 1,
            'session_id': session_id,
            'code_snippet': code_snippet,
            'static_tool_analysis': static_tool_analysis or 'None',
            'model_choice': model_choice,
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'completed_at': None
        }
        
        self.sessions[session_id] = session
        print(f"📝 Created session {session_id} with {len(code_snippet)} characters of code")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def update_session_status(self, session_id: str, status: str):
        """Update session status"""
        if session_id in self.sessions:
            self.sessions[session_id]['status'] = status
            self.sessions[session_id]['updated_at'] = datetime.now().isoformat()
            
            if status == 'completed':
                self.sessions[session_id]['completed_at'] = datetime.now().isoformat()
            
            print(f"📊 Session {session_id} status updated to: {status}")
    
    def create_execution(self, session_id: str, node_id: str, node_type: str, input_data: Dict) -> int:
        """Create a new execution record"""
        self.execution_counter += 1
        execution_id = self.execution_counter
        
        execution = {
            'id': execution_id,
            'session_id': session_id,
            'node_id': node_id,
            'node_type': node_type,
            'input_data': json.dumps(input_data),
            'output_data': None,
            'status': 'running',
            'created_at': datetime.now().isoformat(),
            'execution_time': None,
            'error_message': None
        }
        
        self.executions[execution_id] = execution
        print(f"⚡ Created execution {execution_id} for node {node_type}")
        return execution_id
    
    def update_execution(self, execution_id: int, output_data: Dict = None, status: str = None, 
                        execution_time: float = None, error_message: str = None):
        """Update execution record"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            
            if output_data is not None:
                execution['output_data'] = json.dumps(output_data)
            if status is not None:
                execution['status'] = status
            if execution_time is not None:
                execution['execution_time'] = execution_time
            if error_message is not None:
                execution['error_message'] = error_message
            
            print(f"📈 Updated execution {execution_id} - status: {status}")
    
    def get_executions_for_session(self, session_id: str) -> List[Dict]:
        """Get all executions for a session"""
        executions = []
        for execution in self.executions.values():
            if execution['session_id'] == session_id:
                executions.append(execution)
        
        # Sort by creation time
        executions.sort(key=lambda x: x['created_at'])
        return executions
    
    def save_result(self, session_id: str, findings: List[Dict], full_report: str = None, 
                   model_used: str = None, execution_summary: Dict = None):
        """Save final audit results with new findings format"""
        
        # Validate findings format
        validated_findings = self._validate_findings_format(findings)
        
        result = {
            'session_id': session_id,
            'findings': validated_findings,
            'finding_number': len(validated_findings),
            'full_report': full_report or '',
            'model_used': model_used or 'unknown',
            'execution_summary': execution_summary or {},
            'created_at': datetime.now().isoformat(),
            'severity_breakdown': self._calculate_severity_breakdown(validated_findings)
        }
        
        self.results[session_id] = result
        print(f"💾 Saved results for session {session_id} with {len(validated_findings)} findings")
        
        # Update session status
        self.update_session_status(session_id, 'completed')
        
        return result
    
    def get_result(self, session_id: str) -> Optional[Dict]:
        """Get audit results for a session"""
        return self.results.get(session_id)
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions"""
        sessions = list(self.sessions.values())
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        return sessions
    
    def get_session_progress(self, session_id: str) -> Dict:
        """Calculate session progress based on executions"""
        executions = self.get_executions_for_session(session_id)
        
        if not executions:
            return {
                'total_steps': 0,
                'completed_steps': 0,
                'current_step': 0,
                'progress_percentage': 0,
                'status': 'not_started'
            }
        
        total_steps = len(executions)
        completed_steps = len([e for e in executions if e['status'] == 'completed'])
        failed_steps = len([e for e in executions if e['status'] == 'failed'])
        running_steps = len([e for e in executions if e['status'] == 'running'])
        
        # Calculate progress
        progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Determine overall status
        if failed_steps > 0:
            status = 'failed'
        elif running_steps > 0:
            status = 'running'
        elif completed_steps == total_steps:
            status = 'completed'
        else:
            status = 'in_progress'
        
        return {
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'running_steps': running_steps,
            'current_step': completed_steps + 1 if completed_steps < total_steps else total_steps,
            'progress_percentage': round(progress_percentage, 1),
            'status': status
        }
    
    def _validate_findings_format(self, findings: List[Dict]) -> List[Dict]:
        """Validate and ensure findings follow the correct format"""
        validated_findings = []
        
        for finding in findings:
            if isinstance(finding, dict):
                # Ensure all required fields exist with correct names and types
                validated_finding = {
                    'Issue': str(finding.get('Issue', finding.get('issue', 'Unknown Issue'))),
                    'Severity': str(finding.get('Severity', finding.get('severity', 'Medium'))),
                    'Description': str(finding.get('Description', finding.get('description', 'No description available'))),
                    'Impact': str(finding.get('Impact', finding.get('impact', 'Impact not specified'))),
                    'Location': str(finding.get('Location', finding.get('location', 'Location not specified')))
                }
                
                # Validate severity values
                if validated_finding['Severity'] not in ['High', 'Medium', 'Low']:
                    validated_finding['Severity'] = 'Medium'
                
                # Ensure description mentions source analysis
                if 'appears in' not in validated_finding['Description']:
                    validated_finding['Description'] += ' This issue appears in the auditing result.'
                
                validated_findings.append(validated_finding)
        
        return validated_findings
    
    def _calculate_severity_breakdown(self, findings: List[Dict]) -> Dict:
        """Calculate breakdown of findings by severity"""
        breakdown = {'High': 0, 'Medium': 0, 'Low': 0}
        
        for finding in findings:
            severity = finding.get('Severity', 'Medium')
            if severity in breakdown:
                breakdown[severity] += 1
        
        return breakdown
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        return {
            'total_sessions': len(self.sessions),
            'total_executions': len(self.executions),
            'total_results': len(self.results),
            'completed_sessions': len([s for s in self.sessions.values() if s['status'] == 'completed']),
            'running_sessions': len([s for s in self.sessions.values() if s['status'] == 'running']),
            'failed_sessions': len([s for s in self.sessions.values() if s['status'] == 'failed'])
        }

# Global storage instance
storage = InMemoryStorage()
print("🗄️ In-memory storage initialized successfully")

