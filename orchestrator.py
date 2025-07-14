from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import os

class Orchestrator:
    def __init__(self):
        self.graph = StateGraph()

    def decompose_task(self, task: Dict) -> Dict:
        # Decompose into sub-tasks (e.g., build, test, deploy)
        sub_tasks = {
            "build": "Develop code per goals",
            "test": "Run tests and fix bugs in loop",
            "deploy": "Route to live environment"
        }
        return sub_tasks

    def assign_roles(self, sub_tasks: Dict) -> Crew:
        dev_agent = Agent(role="Dev Expert", goal="Build and code", tools=["cursor_command"])
        tester_agent = Agent(role="Tester", goal="Test and fix in loops", tools=["cursor_command"])
        deployer_agent = Agent(role="Deployer", goal="Route to live", tools=["ssh_deploy"])

        tasks = [Task(description=t, agent=a) for t, a in zip(sub_tasks.values(), [dev_agent, tester_agent, deployer_agent])]
        crew = Crew(agents=[dev_agent, tester_agent, deployer_agent], tasks=tasks)
        return crew

    def run_flow(self, input_task: Dict) -> Any:
        sub_tasks = self.decompose_task(input_task)
        crew = self.assign_roles(sub_tasks)
        self.graph.add_node("dev", crew.agents[0].execute_task)
        self.graph.add_node("tester", crew.agents[1].execute_task)
        self.graph.add_node("deployer", crew.agents[2].execute_task)
        self.graph.add_edge("dev", "tester")
        self.graph.add_conditional_edges("tester", lambda x: "dev" if x['test_fail'] else "deployer")
        self.graph.add_edge("deployer", END)
        return self.graph.compile().invoke(input_task)