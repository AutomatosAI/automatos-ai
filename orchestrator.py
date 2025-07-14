from crewai import Agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any
import os
import requests
from context_manager import ContextManager  # Import for RAG
import os
from dotenv import load_dotenv
load_dotenv()  # Loads .env if present

class State(TypedDict):
    input_task: Dict
    sub_tasks: Dict
    current_result: str
    test_fail: bool

class Orchestrator:
    def __init__(self):
        self.graph = StateGraph(state_schema=State)
        self.mcp_url = os.getenv("MCP_URL", "https://mcp.xplaincrypto.ai/execute")  # Domain for prod
        self.api_key = os.getenv("API_KEY", "your_secure_key")  # For auth

    def decompose_task(self, task: Dict) -> Dict:
        return {
            "build": "Develop code per goals",
            "test": "Run tests and fix bugs in loop",
            "deploy": "Route to live environment"
        }

    def cursor_tool(self, cmd: str) -> str:
        try:
            headers = {"X-API-Key": self.api_key}
            response = requests.post(self.mcp_url, json={"command": cmd}, timeout=10, verify=True, headers=headers)
            response.raise_for_status()
            return response.json().get('stdout', '')
        except Exception as e:
            return f"Tool error: {str(e)}"

    def deploy_tool(self, cmd: str) -> str:
        return self.cursor_tool(f"ssh root@your-droplet-ip '{cmd}'")  # Customize IP/user

    def context_node(self, state: State) -> Dict[str, Any]:
        cm = ContextManager()
        query = state['input_task'].get('task', '')
        context = cm.rag_retrieve(query)
        state['current_result'] += f"Context: {context}\n"
        return state

    def dev_node(self, state: State) -> Dict[str, Any]:
        cmd = "ls -la"  # Example; replace with "cursor --generate code.py"
        result = self.cursor_tool(cmd)
        return {"current_result": state["current_result"] + result, "test_fail": False}

    def tester_node(self, state: State) -> Dict[str, Any]:
        for attempt in range(3):
            cmd = "echo 'Testing attempt'"  # e.g., "pytest"
            result = self.cursor_tool(cmd)
            state["current_result"] += f" Attempt {attempt+1}: {result}"
            if "failed" not in result.lower():
                return {"test_fail": False}
        return {"test_fail": True}

    def deployer_node(self, state: State) -> Dict[str, Any]:
        cmd = "echo 'Deploying...'"  # e.g., "docker compose up -d"
        result = self.deploy_tool(cmd)
        return {"current_result": state["current_result"] + f" Deployed: {result}"}

    def run_flow(self, input_task: Dict) -> Any:
        sub_tasks = self.decompose_task(input_task)
        dev_agent = Agent(role="Dev Expert", goal="Build and code", tools=[self.cursor_tool])
        tester_agent = Agent(role="Tester", goal="Test and fix in loops", tools=[self.cursor_tool])
        deployer_agent = Agent(role="Deployer", goal="Route to live", tools=[self.deploy_tool])

        self.graph.add_node("context", self.context_node)
        self.graph.add_node("dev", self.dev_node)
        self.graph.add_node("tester", self.tester_node)
        self.graph.add_node("deployer", self.deployer_node)
        self.graph.set_entry_point("context")
        self.graph.add_edge("context", "dev")
        self.graph.add_edge("dev", "tester")
        self.graph.add_conditional_edges("tester", lambda state: "dev" if state['test_fail'] else "deployer")
        self.graph.add_edge("deployer", END)

        compiled = self.graph.compile()
        initial_state = {"input_task": input_task, "sub_tasks": sub_tasks, "current_result": "", "test_fail": False}
        return compiled.invoke(initial_state)

if __name__ == "__main__":
    orchestrator = Orchestrator()
    result = orchestrator.run_flow({"task": "Test DevOps Pipeline"})
    print(result)