import json
import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class WorkflowNode(ABC):
    """Base class for all workflow nodes"""
    
    def __init__(self, node_id: str, node_type: str, model_manager=None):
        self.node_id = node_id
        self.node_type = node_type
        self.model_manager = model_manager
    
    @abstractmethod
    def execute_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node synchronously"""
        pass

class InitialAnalysisNode(WorkflowNode):
    """Initial analysis of the smart contract"""
    
    def execute_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform initial analysis of the smart contract"""
        try:
            code_snippet = context.get('code_snippet', '')
            static_tool = context.get('static_tool', '')
            model_choice = context.get('model_choice', 'auto')
            
            # Create prompt for initial analysis
            prompt = f"""
            As a smart contract security auditor, analyze the following Solidity smart contract code and provide an initial comprehensive assessment:

            Smart Contract Code:
            {code_snippet}

            Static Analysis Results (if available):
            {static_tool}

            Please provide a detailed initial analysis including:
            1. Contract Overview and Purpose
               - What does this contract do?
               - What is its main functionality?
               - What blockchain standards does it implement (ERC-20, ERC-721, etc.)?

            2. Architecture Analysis
               - Key components and functions
               - State variables and their purposes
               - Access control mechanisms
               - Inheritance structure (if any)

            3. Initial Security Observations
               - Obvious security patterns or anti-patterns
               - Access control implementation
               - Input validation approaches
               - External call patterns

            4. Potential Areas of Concern
               - Functions that handle value transfers
               - External interactions
               - State-changing functions
               - Areas requiring deeper analysis

            5. Code Quality Assessment
               - Code organization and readability
               - Use of best practices
               - Documentation quality

            Format your response as a structured analysis with clear sections.
            """
            
            # Call AI model
            response = self.model_manager.call_model_sync(prompt)
            
            return {
                'analysis': response,
                'code_snippet': code_snippet,
                'static_tool': static_tool,
                'model_used': model_choice
            }
            
        except Exception as e:
            logger.error(f"Initial analysis failed: {str(e)}")
            return {
                'analysis': f"Initial analysis failed: {str(e)}",
                'code_snippet': context.get('code_snippet', ''),
                'static_tool': context.get('static_tool', ''),
                'model_used': context.get('model_choice', 'unknown')
            }

class AuditPlanNode(WorkflowNode):
    """Create detailed audit plan"""
    
    def execute_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive audit plan"""
        try:
            initial_analysis = context.get('1728096797608', {}).get('analysis', '')
            code_snippet = context.get('code_snippet', '')
            
            prompt = f"""
            Based on the initial analysis, create a comprehensive and systematic audit plan for this smart contract:

            Initial Analysis:
            {initial_analysis}

            Smart Contract Code:
            {code_snippet}

            Create a detailed audit plan covering these areas:

            1. Security Vulnerability Assessment
               - Reentrancy vulnerabilities
               - Integer overflow/underflow
               - Access control issues
               - Front-running vulnerabilities
               - Timestamp dependence
               - Gas limit and loops
               - Unchecked external calls

            2. Business Logic Validation
               - Function logic correctness
               - State transition validation
               - Edge case handling
               - Input validation completeness

            3. Gas Optimization Analysis
               - Expensive operations identification
               - Storage vs memory usage
               - Loop optimization opportunities
               - Function visibility optimization

            4. Standards Compliance
               - ERC standard implementation correctness
               - Interface compliance
               - Event emission standards

            5. Specific Test Scenarios
               - Critical function testing
               - Attack vector simulations
               - Boundary condition tests
               - Integration testing points

            Format the audit plan as numbered tasks that can be executed individually.
            Each task should be specific and actionable.
            """
            
            response = self.model_manager.call_model_sync(prompt)
            
            return {
                'audit_plan': response,
                'tasks': self._extract_tasks(response)
            }
            
        except Exception as e:
            logger.error(f"Audit plan creation failed: {str(e)}")
            return {
                'audit_plan': f"Audit plan creation failed: {str(e)}",
                'tasks': []
            }
    
    def _extract_tasks(self, audit_plan: str) -> List[str]:
        """Extract individual tasks from audit plan"""
        tasks = []
        lines = audit_plan.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up the task text
                task = line.lstrip('0123456789.-• ').strip()
                if task and len(task) > 10:  # Ensure meaningful tasks
                    tasks.append(task)
        return tasks[:12]  # Limit to 12 tasks for manageable execution

class ParameterExtractorNode(WorkflowNode):
    """Extract and format parameters for iteration"""
    
    def execute_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for detailed audit tasks"""
        try:
            audit_plan = context.get('1728113785964', {}).get('audit_plan', '')
            tasks = context.get('1728113785964', {}).get('tasks', [])
            
            # Format tasks for iteration with priorities
            formatted_tasks = []
            priority_keywords = {
                'high': ['reentrancy', 'overflow', 'underflow', 'access control', 'external call'],
                'medium': ['gas', 'optimization', 'validation', 'standard'],
                'low': ['documentation', 'style', 'naming']
            }
            
            for i, task in enumerate(tasks):
                # Determine priority based on keywords
                priority = 'medium'  # default
                task_lower = task.lower()
                for p, keywords in priority_keywords.items():
                    if any(keyword in task_lower for keyword in keywords):
                        priority = p
                        break
                
                formatted_tasks.append({
                    'id': i + 1,
                    'task': task,
                    'priority': priority,
                    'status': 'pending'
                })
            
            # Sort by priority (high first)
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            formatted_tasks.sort(key=lambda x: priority_order.get(x['priority'], 1))
            
            return {
                'tasks': formatted_tasks,
                'total_tasks': len(formatted_tasks),
                'current_task': 0,
                'high_priority_count': len([t for t in formatted_tasks if t['priority'] == 'high']),
                'medium_priority_count': len([t for t in formatted_tasks if t['priority'] == 'medium']),
                'low_priority_count': len([t for t in formatted_tasks if t['priority'] == 'low'])
            }
            
        except Exception as e:
            logger.error(f"Parameter extraction failed: {str(e)}")
            return {
                'tasks': [],
                'total_tasks': 0,
                'current_task': 0
            }

class IterationNode(WorkflowNode):
    """Execute individual audit tasks"""
    
    def execute_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all audit tasks"""
        try:
            tasks = context.get('1728114018925', {}).get('tasks', [])
            code_snippet = context.get('code_snippet', '')
            results = []
            
            for task in tasks:
                try:
                    prompt = f"""
                    As a smart contract security auditor, perform the following specific audit task on this Solidity smart contract:

                    Audit Task: {task.get('task', '')}
                    Priority: {task.get('priority', 'medium')}

                    Smart Contract Code:
                    {code_snippet}

                    Provide detailed findings for this specific task including:

                    1. Issue Analysis
                       - Is there an issue related to this audit task?
                       - If yes, provide detailed description
                       - If no, explain why this area is secure

                    2. Severity Assessment (if issue found)
                       - Critical: Funds can be stolen or contract can be broken
                       - High: Significant impact on contract functionality
                       - Medium: Moderate impact or potential for exploitation
                       - Low: Minor issues or best practice violations

                    3. Technical Details
                       - Specific code location (function names, line references)
                       - Technical explanation of the issue
                       - Attack vectors or exploitation methods

                    4. Impact Assessment
                       - What could happen if this issue is exploited?
                       - Financial impact potential
                       - Operational impact on the contract

                    5. Recommendations
                       - Specific code changes needed
                       - Best practices to implement
                       - Additional security measures

                    Be thorough and specific in your analysis. If no issues are found, clearly state this and explain the security measures that are correctly implemented.
                    """
                    
                    response = self.model_manager.call_model_sync(prompt)
                    
                    results.append({
                        'task_id': task.get('id'),
                        'task': task.get('task'),
                        'priority': task.get('priority'),
                        'result': response,
                        'status': 'completed'
                    })
                    
                except Exception as e:
                    results.append({
                        'task_id': task.get('id'),
                        'task': task.get('task'),
                        'priority': task.get('priority'),
                        'result': f"Task execution failed: {str(e)}",
                        'status': 'failed'
                    })
            
            return {
                'task_results': results,
                'completed_tasks': len([r for r in results if r['status'] == 'completed']),
                'failed_tasks': len([r for r in results if r['status'] == 'failed']),
                'high_priority_completed': len([r for r in results if r['status'] == 'completed' and r.get('priority') == 'high']),
                'total_high_priority': len([r for r in results if r.get('priority') == 'high'])
            }
            
        except Exception as e:
            logger.error(f"Task iteration failed: {str(e)}")
            return {
                'task_results': [],
                'completed_tasks': 0,
                'failed_tasks': 1
            }

class FormatConverterNode(WorkflowNode):
    """Format final audit report with correct findings structure"""
    
    def execute_sync(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final formatted audit report with proper findings format"""
        try:
            task_results = context.get('1728114855021', {}).get('task_results', [])
            code_snippet = context.get('code_snippet', '')
            static_tool = context.get('static_tool', '')
            model_choice = context.get('model_choice', 'auto')
            
            # Create final report with specific findings format
            prompt = f"""
            Based on the comprehensive audit analysis, create a final professional audit report for this smart contract.

            Smart Contract Code:
            {code_snippet}

            Static Analysis Results:
            {static_tool}

            Audit Results Summary:
            - Total tasks completed: {len([r for r in task_results if r['status'] == 'completed'])}
            - High priority tasks: {len([r for r in task_results if r.get('priority') == 'high'])}

            Detailed Task Results:
            {json.dumps(task_results, indent=2)}

            Create a final report with findings in this EXACT JSON format:

            [
              {{
                "Issue": "A brief title or summary of the issue",
                "Severity": "High / Medium / Low, based on impact and security risk",
                "Description": "Explain the issue and how it relates to both sources. State clearly whether it appears in the knowledge base, the auditing result, or both.",
                "Impact": "Potential risks or negative consequences if present.",
                "Location": "Precise location in the auditing result (e.g., function name, line number) and/or reference in the knowledge base."
              }}
            ]

            IMPORTANT REQUIREMENTS:
            1. Each finding MUST have exactly these 5 fields: Issue, Severity, Description, Impact, Location
            2. Severity MUST be exactly "High", "Medium", or "Low"
            3. Description MUST clearly state whether the issue appears in:
               - The smart contract code only
               - The static analysis results only  
               - Both the smart contract code and static analysis results
               - Neither (if it's a potential issue based on code patterns)
            4. Location MUST specify exact function names, line numbers, or static analysis references
            5. Return ONLY the JSON array of findings, no additional text
            6. If no issues are found, return an empty array: []

            Analyze all the task results and create findings that follow this exact format.
            """
            
            response = self.model_manager.call_model_sync(prompt)
            
            # Try to extract JSON findings from response
            findings = self._extract_json_findings(response)
            
            # If extraction fails, create findings from task results
            if not findings:
                findings = self._create_findings_from_tasks(task_results, code_snippet, static_tool)
            
            # Validate and clean findings format
            validated_findings = self._validate_findings_format(findings)
            
            return {
                'Findings': validated_findings,
                'FindingNumber': len(validated_findings),
                'FullReport': response,
                'ModelUsed': model_choice,
                'CriticalIssues': len([f for f in validated_findings if f.get('Severity') == 'High']),
                'MediumIssues': len([f for f in validated_findings if f.get('Severity') == 'Medium']),
                'LowIssues': len([f for f in validated_findings if f.get('Severity') == 'Low']),
                'ExecutionSummary': {
                    'total_tasks': len(task_results),
                    'completed_tasks': len([r for r in task_results if r['status'] == 'completed']),
                    'failed_tasks': len([r for r in task_results if r['status'] == 'failed'])
                }
            }
            
        except Exception as e:
            logger.error(f"Report formatting failed: {str(e)}")
            # Return error as a finding
            error_finding = {
                'Issue': 'Report Generation Error',
                'Severity': 'High',
                'Description': f'Failed to generate audit report: {str(e)}. This error appears in the auditing result only.',
                'Impact': 'Unable to complete comprehensive audit analysis, potential security issues may be missed.',
                'Location': 'Report Generator - FormatConverterNode'
            }
            
            return {
                'Findings': [error_finding],
                'FindingNumber': 1,
                'FullReport': f'Report generation failed: {str(e)}',
                'ModelUsed': context.get('model_choice', 'unknown'),
                'CriticalIssues': 0,
                'MediumIssues': 0,
                'LowIssues': 0
            }
    
    def _extract_json_findings(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON findings from AI response"""
        try:
            # Try to find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                findings = json.loads(json_str)
                
                # Validate that it's a list of dictionaries
                if isinstance(findings, list):
                    return findings
                    
        except Exception as e:
            logger.warning(f"Failed to extract JSON findings: {e}")
        
        return []
    
    def _create_findings_from_tasks(self, task_results: List[Dict], code_snippet: str, static_tool: str) -> List[Dict[str, Any]]:
        """Create findings from task results if JSON extraction fails"""
        findings = []
        
        for result in task_results:
            if result['status'] == 'completed':
                result_text = result['result']
                priority = result.get('priority', 'medium')
                
                # Look for severity indicators in the result
                severity = 'Medium'  # default
                if any(word in result_text.upper() for word in ['CRITICAL', 'HIGH RISK', 'SEVERE']):
                    severity = 'High'
                elif any(word in result_text.upper() for word in ['LOW RISK', 'MINOR', 'INFORMATIONAL']):
                    severity = 'Low'
                
                # Determine source based on content
                has_static_ref = bool(static_tool and any(word in result_text.lower() for word in ['static', 'analysis', 'tool']))
                has_code_ref = any(word in result_text.lower() for word in ['function', 'line', 'contract', 'variable'])
                
                if has_static_ref and has_code_ref:
                    source_desc = "This issue appears in both the smart contract code and static analysis results."
                elif has_static_ref:
                    source_desc = "This issue appears in the static analysis results only."
                elif has_code_ref:
                    source_desc = "This issue appears in the smart contract code only."
                else:
                    source_desc = "This issue is identified based on code pattern analysis."
                
                finding = {
                    'Issue': f'{priority.title()} Priority Security Issue - {result.get("task", "Unknown")[:50]}...',
                    'Severity': severity,
                    'Description': f'{result_text[:200]}... {source_desc}',
                    'Impact': f'Potential {severity.lower()} impact on contract security and functionality.',
                    'Location': f'Task {result.get("task_id", "unknown")}: {result.get("task", "Unknown task")}'
                }
                
                findings.append(finding)
        
        return findings
    
    def _validate_findings_format(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and ensure findings follow the correct format"""
        validated_findings = []
        
        for finding in findings:
            if isinstance(finding, dict):
                # Ensure all required fields exist with correct names
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
                
                validated_findings.append(validated_finding)
        
        return validated_findings


