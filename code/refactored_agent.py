"""
Refactored IMO Problem Solver Agent using LiteLLM and LangGraph
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Generator
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv(override=True)

# LiteLLM for model abstraction
import litellm
from litellm import completion

# LangGraph for workflow orchestration
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# Add this import for message type checking
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

# ANSI color codes for logging
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configure LiteLLM
litellm.drop_params = True  # Drop unsupported params instead of raising errors
litellm.set_verbose = False  # Set to True for debugging

# --- State Definition ---
class AgentState(TypedDict):
    """State for the IMO problem solver agent"""
    problem_statement: str
    current_solution: Optional[str]
    verification_result: Optional[str]
    is_solution_complete: bool
    is_solution_verified: bool
    error_count: int
    correct_count: int
    iteration_count: int
    conversation_history: Annotated[List[Dict[str, Any]], add_messages]
    additional_prompts: List[str]
    logs: List[str]

# --- Model Client with LiteLLM ---
class ModelClient:
    """Universal model client using LiteLLM"""
    
    def __init__(self, model: str, temperature: float = 0.1, think_tokens: Optional[int] = None):
        self.model = model
        self.temperature = temperature
        self.think_tokens = think_tokens
        
        # Setup Azure-specific configuration if needed
        if "azure" in model.lower():
            self._setup_azure()
    
    def _setup_azure(self):
        """Setup Azure-specific configuration"""
        # Ensure required environment variables are set
        required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Error: Missing required Azure environment variables: {', '.join(missing_vars)}")
            print("\nPlease set the following environment variables:")
            print("export AZURE_OPENAI_API_KEY='your-azure-api-key'")
            print("export AZURE_OPENAI_ENDPOINT='https://your-resource-name.openai.azure.com/'")
            print("export AZURE_OPENAI_API_VERSION='your-api-version'")
            sys.exit(1)
        
        # Store Azure config for later use
        self.azure_config = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
        }
        
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator:
        """
        Generate streaming response using LiteLLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the model
        
        Yields:
            Text chunks from the model
        """
        try:
            # Build generation config
            gen_config = {
                "temperature": self.temperature,
            }
            
            # Add Azure-specific config if using Azure, only way to run liteLLM with Azure, tried with many options
            if "azure" in self.model.lower() and hasattr(self, 'azure_config'):
                gen_config.update(self.azure_config)
            
            # Add Gemini-specific thinking process config to sync with the original paper
            if self.think_tokens and "gemini" in self.model.lower():
                gen_config["response_modalities"] = ["TEXT"]
                gen_config["response_schema"] = {
                    "thinking_process": {
                        "type": "STRING",
                        "maxOutputTokens": self.think_tokens
                    },
                    "output": {
                        "type": "STRING"
                    }
                }
            
            # Merge with additional kwargs
            gen_config.update(kwargs)
            
            response = completion(
                model=self.model,
                messages=messages,
                stream=True,
                **gen_config
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error in model generation: {e}")
            raise

# --- Prompts Manager ---
class PromptsManager:
    """Manages all prompts used in the system"""
    
    def __init__(self, prompts_dir: str = "code/prompts"):
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory {prompts_dir} does not exist")
    
    def load_prompt(self, prompt_name: str) -> str:
        """Load a prompt from file"""
        prompt_path = self.prompts_dir / f"{prompt_name}.md"
        if not prompt_path.exists():
            raise ValueError(f"Prompt file {prompt_path} does not exist")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def get_initial_solution_prompt(self) -> str:
        return self.load_prompt("initial_solution_prompt")
    
    def get_self_improvement_prompt(self) -> str:
        return self.load_prompt("self_improvement_prompt")
    
    def get_verification_prompt(self) -> str:
        return self.load_prompt("verification_prompt")
    
    def get_correction_prompt(self) -> str:
        return self.load_prompt("correction_prompt")

# --- Node Functions for LangGraph ---
class IMOSolverNodes:
    """Node functions for the IMO solver workflow"""
    
    def __init__(self, model_client: ModelClient, prompts_manager: PromptsManager):
        self.model = model_client
        self.prompts = prompts_manager

    def _log_node_entry(self, node_name: str, state: AgentState):
        """Logs the entry into a node."""
        print(f"\n{bcolors.HEADER}---► Entering: {node_name}{bcolors.ENDC}")
        # Optionally log parts of the state, e.g., iteration count
        if "iteration_count" in state:
            print(f"{bcolors.OKCYAN}    Iteration: {state['iteration_count']}{bcolors.ENDC}")

    def _log_node_exit(self, node_name: str, result: Dict[str, Any]):
        """Logs the exit from a node."""
        print(f"{bcolors.OKBLUE}---◄ Exiting: {node_name}{bcolors.ENDC}")
        # Log key results for clarity
        for key, value in result.items():
            if key not in ["conversation_history", "logs"]: # Avoid printing long lists
                print(f"{bcolors.OKGREEN}    - {key}: {str(value)[:200]}...{bcolors.ENDC}")
    
    def _convert_messages_to_dict(self, messages: List[BaseMessage | dict]) -> List[Dict[str, str]]:
        """Converts a list of LangChain messages or dicts to a list of dicts for LiteLLM."""
        output = []
        for m in messages:
            role = ""
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            
            if role:
                output.append({"role": role, "content": str(m.content)})
            elif isinstance(m, dict) and "role" in m and "content" in m:
                output.append(m)
        return output

    def _stream_and_collect(self, messages: List[Any], prefix: str = "") -> str:
        """Stream response and collect the full text"""
        # Convert messages to dicts right before calling the model
        dict_messages = self._convert_messages_to_dict(messages)
        
        print(prefix, end='', flush=True)
        solution_parts = []
        for chunk in self.model.generate_stream(dict_messages):
            print(chunk, end='', flush=True)
            solution_parts.append(chunk)
        print()  # New line after streaming
        return ''.join(solution_parts)
    
    def initial_solution_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate initial solution"""
        node_name = "Initial Solution"
        self._log_node_entry(node_name, state)
        
        messages = [
            {"role": "system", "content": self.prompts.get_initial_solution_prompt()},
            {"role": "user", "content": state["problem_statement"]}
        ]
        
        # Add additional prompts if provided
        for prompt in state.get("additional_prompts", []):
            messages.append({"role": "user", "content": prompt})
        
        # Generate solution with streaming
        solution = self._stream_and_collect(messages, f"{bcolors.BOLD}Generating initial solution...{bcolors.ENDC}\n")
        
        result = {
            "current_solution": solution,
            "conversation_history": messages + [{"role": "assistant", "content": solution}],
            "logs": state.get("logs", []) + [f"Generated initial solution at {datetime.now()}"]
        }
        self._log_node_exit(node_name, result)
        return result
    
    def self_improvement_node(self, state: AgentState) -> Dict[str, Any]:
        """Self-improve the solution"""
        node_name = "Self Improvement"
        self._log_node_entry(node_name, state)
        
        messages = state["conversation_history"] + [
            {"role": "user", "content": self.prompts.get_self_improvement_prompt()}
        ]
        
        # Add additional prompts for reinforcement
        for prompt in state.get("additional_prompts", []):
            messages.append({"role": "user", "content": f"Reminder of additional instructions: {prompt}"})

        # Generate improved solution with streaming
        improved_solution = self._stream_and_collect(messages, f"{bcolors.BOLD}Improving solution...{bcolors.ENDC}\n")
        
        result = {
            "current_solution": improved_solution,
            "conversation_history": messages + [{"role": "assistant", "content": improved_solution}],
            "logs": state.get("logs", []) + [f"Self-improved solution at {datetime.now()}"]
        }
        self._log_node_exit(node_name, result)
        return result
    
    def check_completeness_node(self, state: AgentState) -> Dict[str, Any]:
        """Check if solution claims to be complete"""
        node_name = "Completeness Check"
        self._log_node_entry(node_name, state)
        
        check_prompt = f"""
Is the following text claiming that the solution is complete?
{state["current_solution"]}
Response in exactly "yes" or "no". No other words.
"""
        
        messages = [{"role": "user", "content": check_prompt}]
        response = self._stream_and_collect(messages, f"{bcolors.BOLD}Checking completeness: {bcolors.ENDC}")
        
        is_complete = "yes" in response.lower()
        
        result = {
            "is_solution_complete": is_complete,
            "logs": state.get("logs", []) + [f"Completeness check: {is_complete} at {datetime.now()}"]
        }
        self._log_node_exit(node_name, result)
        return result
    
    def verification_node(self, state: AgentState) -> Dict[str, Any]:
        """Verify the solution"""
        node_name = "Verification"
        self._log_node_entry(node_name, state)
        
        # This is the content from the original agent's `verification_remider`
        verification_reminder_text = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""

        verification_prompt = f"""
### Problem ###
{state["problem_statement"]}

### Solution ###
{state["current_solution"]}

{verification_reminder_text}
"""
        
        messages = [
            {"role": "system", "content": self.prompts.get_verification_prompt()},
            {"role": "user", "content": verification_prompt}
        ]

        # Add additional prompts for context
        for prompt in state.get("additional_prompts", []):
            messages.append({"role": "user", "content": f"Reminder of additional instructions for verification: {prompt}"})
        
        # Generate verification with streaming
        verification_result = self._stream_and_collect(messages, f"{bcolors.BOLD}Verifying solution...{bcolors.ENDC}\n")
        
        # Check if verification passed
        check_messages = [
            {"role": "user", "content": f"Does this verification indicate the solution is correct? Answer yes or no:\n{verification_result}"}
        ]
        check_response = self._stream_and_collect(check_messages, f"{bcolors.BOLD}Checking verification result: {bcolors.ENDC}")
        
        is_verified = "yes" in check_response.lower()
        
        # Update counts
        correct_count = state.get("correct_count", 0) + (1 if is_verified else 0)
        error_count = state.get("error_count", 0) + (0 if is_verified else 1)
        
        result = {
            "verification_result": verification_result,
            "is_solution_verified": is_verified,
            "correct_count": correct_count if is_verified else 0,
            "error_count": error_count if not is_verified else 0,
            "logs": state.get("logs", []) + [f"Verification: {is_verified} at {datetime.now()}"]
        }
        self._log_node_exit(node_name, result)
        return result
    
    def correction_node(self, state: AgentState) -> Dict[str, Any]:
        """Correct the solution based on verification feedback"""
        node_name = "Correction"
        self._log_node_entry(node_name, state)
        
        messages = [
            {"role": "system", "content": self.prompts.get_initial_solution_prompt()},
            {"role": "user", "content": state["problem_statement"]},
            {"role": "assistant", "content": state["current_solution"]},
            {"role": "user", "content": self.prompts.get_correction_prompt()},
            {"role": "user", "content": state["verification_result"]}
        ]

        # Add additional prompts for context
        for prompt in state.get("additional_prompts", []):
            messages.append({"role": "user", "content": f"Reminder of additional instructions for correction: {prompt}"})
        
        # Generate corrected solution with streaming
        corrected_solution = self._stream_and_collect(messages, f"{bcolors.BOLD}Correcting solution based on feedback...{bcolors.ENDC}\n")
        
        iteration_count = state.get("iteration_count", 0) + 1
        
        result = {
            "current_solution": corrected_solution,
            "conversation_history": messages + [{"role": "assistant", "content": corrected_solution}],
            "iteration_count": iteration_count,
            "logs": state.get("logs", []) + [f"Corrected solution (iteration {iteration_count}) at {datetime.now()}"]
        }
        self._log_node_exit(node_name, result)
        return result

# --- Workflow Definition ---
def create_imo_solver_workflow(model_client: ModelClient, prompts_manager: PromptsManager):
    """Create the LangGraph workflow for IMO problem solving"""
    
    # Initialize nodes
    nodes = IMOSolverNodes(model_client, prompts_manager)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initial_solution", nodes.initial_solution_node)
    workflow.add_node("self_improvement", nodes.self_improvement_node)
    workflow.add_node("check_completeness", nodes.check_completeness_node)
    workflow.add_node("verification", nodes.verification_node)
    workflow.add_node("correction", nodes.correction_node)
    
    # Define edges
    workflow.set_entry_point("initial_solution")
    workflow.add_edge("initial_solution", "self_improvement")
    workflow.add_edge("self_improvement", "check_completeness")
    
    # Conditional edges
    def should_continue_after_completeness(state: AgentState) -> str:
        if state["is_solution_complete"]:
            return "verification"
        else:
            return END
    
    def should_continue_after_verification(state: AgentState) -> str:
        if state["is_solution_verified"]:
            if state["correct_count"] >= 5:
                return END
            else:
                return "verification"  # Verify again
        else:
            if state["error_count"] >= 10:
                return END
            else:
                return "correction"
    
    workflow.add_conditional_edges(
        "check_completeness",
        should_continue_after_completeness,
        {
            "verification": "verification",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "verification",
        should_continue_after_verification,
        {
            "verification": "verification",
            "correction": "correction",
            END: END
        }
    )
    
    workflow.add_edge("correction", "check_completeness")

    # Compile the graph to get the runnable app
    app = workflow.compile()
    
    # Print the graph structure in Mermaid format
    print("\n--- LangGraph Mermaid Diagram (for docs/viz) ---")
    print(app.get_graph().draw_mermaid())
    app.get_graph().draw_mermaid_png(output_file_path="docs/imo_solver_workflow.png")
    # app.get_graph().draw_mermaid().write_to_file("docs/imo_solver_workflow.mmd")
    print("------------------------------------------------\n") 

    return app

# --- Main Agent Class ---
class IMOSolverAgent:
    """Main agent class for solving IMO problems"""
    
    def __init__(self, model: str, temperature: float = 0.1, think_tokens: Optional[int] = None):
        self.model_client = ModelClient(model, temperature, think_tokens)
        self.prompts_manager = PromptsManager()
        self.workflow = create_imo_solver_workflow(self.model_client, self.prompts_manager)
    
    def solve(self, problem_statement: str, additional_prompts: List[str] = None) -> Optional[str]:
        """
        Solve an IMO problem
        
        Args:
            problem_statement: The problem to solve
            additional_prompts: Additional guidance for the solver
        
        Returns:
            The final solution if successful, None otherwise
        """
        # Initialize state
        initial_state = {
            "problem_statement": problem_statement,
            "current_solution": None,
            "verification_result": None,
            "is_solution_complete": False,
            "is_solution_verified": False,
            "error_count": 0,
            "correct_count": 0,
            "iteration_count": 0,
            "conversation_history": [],
            "additional_prompts": additional_prompts or [],
            "logs": []
        }
        
        # Run the workflow
        print("Starting IMO problem solver workflow...")
        final_state = self.workflow.invoke(initial_state)
        
        # Print logs
        print("\n--- Execution Logs ---")
        for log in final_state.get("logs", []):
            print(f"  {log}")
        
        # Return the final solution if successful
        if final_state["is_solution_verified"] and final_state["correct_count"] >= 5:
            return final_state["current_solution"]
        else:
            print("\nFailed to find a verified solution.")
            return None

# --- Utility Functions ---
def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    if log_file:
        # Redirect stdout to both console and file
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        log_handle = open(log_file, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_handle)
        sys.stderr = Tee(sys.stderr, log_handle)

# --- Main Entry Point ---
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Refactored IMO Problem Solver Agent')
    parser.add_argument('problem_file', nargs='?', default='problem_statement.txt',
                       help='Path to the problem statement file')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Model to use (e.g., gemini/gemini-2.5-pro, azure/gpt-4.1, claude-3-opus)')
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Temperature for generation')
    parser.add_argument('--think-tokens', type=int, default=None,
                       help='Maximum tokens for thinking process (Gemini only)')
    parser.add_argument('--log', '-l', type=str, help='Path to log file')
    parser.add_argument('--additional-prompts', '-p', type=str,
                       help='Additional prompts separated by |')
    parser.add_argument('--max-runs', '-r', type=int, default=10,
                       help='Maximum number of runs')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log)
    
    # Read problem statement
    try:
        with open(args.problem_file, 'r', encoding='utf-8') as f:
            problem_statement = f.read()
    except Exception as e:
        print(f"Error reading problem file: {e}")
        sys.exit(1)
    
    # Parse additional prompts
    additional_prompts = []
    if args.additional_prompts:
        additional_prompts = args.additional_prompts.split('|')
    
    # Create and run agent
    agent = IMOSolverAgent(
        model=args.model, 
        temperature=args.temperature,
        think_tokens=args.think_tokens
    )
    
    for run in range(args.max_runs):
        print(f"\n\n{'='*60}")
        print(f"Run {run + 1} of {args.max_runs}")
        print(f"{'='*60}")
        
        try:
            solution = agent.solve(problem_statement, additional_prompts)
            if solution:
                print(f"\n\nSuccessfully found a verified solution in run {run + 1}!")
                print("\n--- Final Solution ---")
                print(solution)
                break
        except Exception as e:
            print(f"Error in run {run + 1}: {e}")
            continue
    else:
        print(f"\n\nFailed to find a solution after {args.max_runs} runs.")

if __name__ == "__main__":
    main()