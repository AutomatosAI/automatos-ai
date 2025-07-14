from crewai import Agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any
import os
import requests
from context_manager import ContextManager
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    input_task: Dict
    sub_tasks: Dict
    current_result: str
    test_fail: bool

class Orchestrator:
    def __init__(self):
        self.graph = StateGraph(state_schema=State)
        self.mcp_url = os.getenv("MCP_URL", "https://mcp.xplaincrypto.ai/execute")
        self.api_key = os.getenv("API_KEY")

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
        return self.cursor_tool(cmd)

    def context_node(self, state: State) -> Dict[str, Any]:
        cm = ContextManager()
        query = state['input_task'].get('task', '')
        context = cm.rag_retrieve(query)
        state['current_result'] += f"Context: {context}\n"
        return state

    def dev_node(self, state: State) -> Dict[str, Any]:
        github_url = state['input_task'].get('github_url', '')
        token = os.getenv('GITHUB_TOKEN', '')
        clone_cmd = f"git clone https://{token}@{github_url.replace('https://', '')} /app/project"
        self.cursor_tool(clone_cmd)
        # Basic mock build (replace with DeepAgent later)
        build_cmd = "echo 'Mock code generated for task' >> /app/project/main.py"
        result = self.cursor_tool(build_cmd)
        return {"current_result": state["current_result"] + result, "test_fail": False}

    def tester_node(self, state: State) -> Dict[str, Any]:
        for attempt in range(3):
            test_cmd = "python /app/project/tests.py"  # Assume tests.py exists
            result = self.cursor_tool(test_cmd)
            state["current_result"] += f" Attempt {attempt+1}: {result}"
            if "failed" not in result.lower():
                return {"test_fail": False}
        return {"test_fail": True}

    def deployer_node(self, state: State) -> Dict[str, Any]:
        token = os.getenv('GITHUB_TOKEN', '')
        cmd = f"cd /app/project && git add . && git commit -m 'AI Build for task' && git push origin main"
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
    result = orchestrator.run_flow({"task": "Test DevOps Pipeline", "github_url": "https://github.com/youruser/test-devops-repo.git"})
    print(result)