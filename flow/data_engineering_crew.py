from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, crew, task
import sys
from pathlib import Path
from dotenv import load_dotenv
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.models import (
    get_advanced_reasoning_model,
    get_high_reasoning_model,
    get_low_reasoning_model,
    get_medium_reasoning_model,
)

from utils.custom_tools import AgenticPDFSearchTool, AgenticExcelSearchTool
from utils.utils import print_directory_tree_treelib, load_configs

load_dotenv(".env")


@CrewBase
class DataEngineeringCrew:
    """Data Engineering Crew that handle Agentic RAG"""

    # agents: List[BaseAgent]
    # tasks: List[Task]

    def __init__(self):
        self.configs = load_configs("data_engineering")
        self.agents_configs = self.configs["agents"]
        self.tasks_configs = self.configs["tasks"]

        self.knowledge_source = "knowledge/"

        self.pdf_search_tool = AgenticPDFSearchTool(source=self.knowledge_source)
        self.excel_search_tool = AgenticExcelSearchTool(source=self.knowledge_source)
        self.serper_tool = SerperDevTool()

    @agent
    def query_agent(self) -> Agent:
        return Agent(
            config=self.agents_configs["query_agent"],
            verbose=True,
            llm=get_high_reasoning_model(
                model=os.getenv("OLLAMA_MODEL_NAME"),
                base_url=os.getenv("OLLAMA_API_BASE"),
                api_key="",
            ),
            reasoning=False,
        )

    @task
    def resource_query(self):
        return Task(
            config=self.tasks_configs["resource_query"],
            agent=self.query_agent(),
            create_directory=True,
            output_file="output/prompt-to-questions.md",
        )

    @agent
    def resource_pulling_agent(self) -> Agent:
        return Agent(
            config=self.agents_configs["resource_pulling_agent"],
            verbose=True,
            llm=get_advanced_reasoning_model(),
            # llm=get_high_reasoning_model(
            #     model=os.getenv("OLLAMA_MODEL_NAME"),
            #     base_url=os.getenv("OLLAMA_API_BASE"),
            #     api_key="",
            # ),
            reasoning=True,
        )

    @task
    def resource_pulling(self):
        return Task(
            config=self.tasks_configs["resource_pulling"],
            agent=self.resource_pulling_agent(),
            create_directory=True,
            tools=[
                self.pdf_search_tool,
                self.excel_search_tool,
                self.serper_tool,
            ],
            output_file="output/info-detailed-report.md",
            context=[self.resource_query()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.query_agent(), self.resource_pulling_agent()],
            tasks=[
                self.resource_query(),
                self.resource_pulling(),
            ],
            verbose=True,
            # knowledge_source=self.pdf_source,
        )


class DeepResearchInput(BaseModel):
    topic: str = Field(..., description="The topic to perform deep research on.")


class DeepResearchTool(BaseTool):
    name: str = "deep_research_tool"
    description: str = "Tool to perform deep research on given topics."
    args_schema: BaseModel = DeepResearchInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._crew = DataEngineeringCrew().crew()

    def _run(self, topic: str) -> str:
        # Implement deep research logic here
        result = self._crew.kickoff(
            {
                "user_prompt": topic,
                "files": str(print_directory_tree_treelib("knowledge")),
            }
        )
        return result.raw


if __name__ == "__main__":
    crew = DataEngineeringCrew().crew()
    # crew.train(
    #     inputs={
    #         "user_prompt": """
    #         Compare Rexit 2022's total revenue with Kelington berhad's. List out the recent company strategic movements.
    #         """,
    #         "files": str(print_directory_tree_treelib("knowledge")),
    #     },
    #     n_iterations=2,
    #     filename="data_engineering_crew_training.pkl",
    # )
    crew.kickoff(
        inputs={
            "user_prompt": """
            Compare Rexit 2022's total revenue with Kelington berhad's. List out the recent company strategic movements.
            """,
            "files": str(print_directory_tree_treelib("knowledge")),
        },
        
    )
