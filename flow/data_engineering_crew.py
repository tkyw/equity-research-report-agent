from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, crew, task
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.models import (
    get_advanced_reasoning_model,
    get_high_reasoning_model,
    get_low_reasoning_model,
    get_medium_reasoning_model,
)

from utils.custom_tools import AgenticPDFSearchTool
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


    @agent
    def query_agent(self) -> Agent:
        return Agent(
            config=self.agents_configs["query_agent"],
            verbose=True,
            llm=get_high_reasoning_model(),
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
            reasoning=True,
            memory=True,
        )

    @task
    def table_of_content_generation(self):
        return Task(
            config=self.tasks_configs["table_of_content_generation"],
            agent=self.resource_pulling_agent(),
            tools=[self.pdf_search_tool],
            create_directory=True,
            output_file="output/table-of-content.md",
            context=[self.resource_query()],
        )

    @task
    def resource_pulling(self):
        return Task(
            config=self.tasks_configs["resource_pulling"],
            agent=self.resource_pulling_agent(),
            create_directory=True,
            tools=[self.pdf_search_tool],
            output_file="output/info-detailed-report.md",
            context=[self.resource_query(), self.table_of_content_generation()],
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


if __name__ == "__main__":
    crew = DataEngineeringCrew().crew()
    crew.kickoff(
        {
            "user_prompt": """
            Provide a detailed report on the company's core business, financial performance, recent developments, and future outlook.
            """,
            "files": str(print_directory_tree_treelib("knowledge")),
        }
    )