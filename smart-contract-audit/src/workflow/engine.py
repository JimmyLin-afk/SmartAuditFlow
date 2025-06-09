import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.storage.memory import storage
from src.workflow.nodes import (
    InitialAnalysisNode, AuditPlanNode, ParameterExtractorNode,
    IterationNode, FormatConverterNode
)
from src.workflow.models import ModelManager

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """Main workflow execution engine with model selection support"""
    
    def __init__(self, session_id: str, model_choice: str = 'auto'):
        from src.workflow.models import get_model_manager
        
        self.session_id = session_id
        self.model_choice = model_choice
        self.model_manager = get_model_manager()
        self.model_manager.set_preferred_model(model_choice)
        self.nodes = {}
        self.execution_order = []
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Setup workflow nodes and execution order"""
        # Initialize workflow nodes with model manager
        self.nodes = {
            '1728096797608': InitialAnalysisNode('1728096797608', 'llm', self.model_manager),
            '1728113785964': AuditPlanNode('1728113785964', 'llm', self.model_manager),
            '1728114018925': ParameterExtractorNode('1728114018925', 'parameter-extractor', self.model_manager),
            '1728114855021': IterationNode('1728114855021', 'iteration', self.model_manager),
            '1728120734743': FormatConverterNode('1728120734743', 'llm', self.model_manager)
        }
        
        # Define execution order based on the workflow graph
        self.execution_order = [
            '1728096797608',  # Initial Analysis
            '1728113785964',  # Audit Plan
            '1728114018925',  # Parameter Extractor
            '1728114855021',  # Iteration
            '1728120734743'   # Format Converter
        ]
    
    def execute_workflow_sync(self, code_snippet: str, static_tool: str = "") -> Dict[str, Any]:
        """
        Execute the complete workflow synchronously with selected model
        
        Args:
            code_snippet: Smart contract code to audit
            static_tool: Static analysis results
            
        Returns:
            Final audit results
        """
        try:
            # Update session status
            storage.update_session_status(self.session_id, 'running')
            
            # Log model selection
            logger.info(f"Starting workflow for session {self.session_id} with model: {self.model_choice}")
            print(f"🤖 Using AI model: {self.model_choice}")
            
            # Check if model is available
            available_models = self.model_manager.get_available_models()
            if self.model_choice == 'gemini' and not available_models.get('gemini'):
                raise Exception("Gemini model is not available. Please check your GEMINI_API_KEY.")
            elif self.model_choice == 'openai' and not available_models.get('openai'):
                raise Exception("OpenAI model is not available. Please check your OPENAI_API_KEY.")
            elif self.model_choice == 'auto' and not (available_models.get('gemini') or available_models.get('openai')):
                raise Exception("No AI models are available. Please check your API keys.")
            
            # Initialize workflow context
            context = {
                'code_snippet': code_snippet,
                'static_tool': static_tool,
                'session_id': self.session_id,
                'model_choice': self.model_choice
            }
            
            # Execute nodes in order
            for i, node_id in enumerate(self.execution_order):
                node = self.nodes[node_id]
                logger.info(f"Executing node {i+1}/{len(self.execution_order)}: {node_id} ({node.node_type}) with model {self.model_choice}")
                print(f"📋 Step {i+1}/{len(self.execution_order)}: {node.node_type}")
                
                # Create workflow execution record
                execution_id = storage.create_execution(
                    session_id=self.session_id,
                    node_id=node_id,
                    node_type=node.node_type,
                    input_data={'step': i+1, 'total_steps': len(self.execution_order)}
                )
                
                start_time = time.time()
                
                try:
                    # Execute the node synchronously
                    print(f"⚡ Executing {node.node_type}...")
                    result = node.execute_sync(context)
                    execution_time = time.time() - start_time
                    
                    # Update context with result
                    context[node_id] = result
                    
                    # Update execution record
                    storage.update_execution(
                        execution_id=execution_id,
                        output_data={'summary': f'Completed in {execution_time:.2f}s'},
                        status='completed',
                        execution_time=execution_time
                    )
                    
                    logger.info(f"✅ Node {node_id} completed in {execution_time:.2f}s")
                    print(f"✅ {node.node_type} completed in {execution_time:.2f}s")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_msg = str(e)
                    
                    # Update execution record with error
                    storage.update_execution(
                        execution_id=execution_id,
                        status='failed',
                        error_message=error_msg,
                        execution_time=execution_time
                    )
                    
                    logger.error(f"❌ Node {node_id} failed: {error_msg}")
                    print(f"❌ {node.node_type} failed: {error_msg}")
                    
                    # Update session status
                    storage.update_session_status(self.session_id, 'failed')
                    
                    raise Exception(f"Workflow failed at step {i+1} ({node.node_type}): {error_msg}")
            
            # Get final results from the last node
            final_results = context.get('1728120734743', {})
            
            # Add model information to results
            final_results['model_used'] = self.model_choice
            final_results['available_models'] = available_models
            final_results['execution_summary'] = {
                'total_steps': len(self.execution_order),
                'completed_steps': len(self.execution_order),
                'session_id': self.session_id
            }
            
            # Save audit results
            storage.save_result(
                session_id=self.session_id,
                findings=final_results.get('Findings', []),
                full_report=final_results.get('FullReport', ''),
                model_used=self.model_choice,
                execution_summary=final_results.get('ExecutionSummary', {})
            )
            
            # Update session status
            storage.update_session_status(self.session_id, 'completed')
            
            logger.info(f"🎉 Workflow completed successfully for session {self.session_id} using model {self.model_choice}")
            print(f"🎉 Audit completed successfully! Found {final_results.get('FindingNumber', 0)} findings.")
            return final_results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"💥 Workflow execution failed for session {self.session_id}: {error_msg}")
            print(f"💥 Workflow failed: {error_msg}")
            
            # Update session status
            storage.update_session_status(self.session_id, 'failed')
            
            # Save error as result
            error_finding = {
                'Issue': 'Workflow Execution Error',
                'Severity': 'High',
                'Description': f'The audit workflow failed with error: {error_msg}. This error appears in the auditing result only.',
                'Impact': 'Unable to complete security analysis, potential vulnerabilities may remain undetected.',
                'Location': 'Workflow Engine - Main Execution Loop'
            }
            
            storage.save_result(
                session_id=self.session_id,
                findings=[error_finding],
                full_report=f'Workflow execution failed: {error_msg}',
                model_used=self.model_choice,
                execution_summary={'error': True, 'failed_at': 'workflow_execution'}
            )
            
            raise e
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current workflow progress"""
        return storage.get_session_progress(self.session_id)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get workflow execution logs"""
        return storage.get_executions_for_session(self.session_id)

