from crewai.tools import BaseTool
from crewai_tools import PDFSearchTool
from pydantic import BaseModel, Field
import os

class AgenticPDFInput(BaseModel):
    query: str = Field(..., description="The search query to find information in the PDF.")
    pdf_path: str = Field(..., description="The path to the PDF document to be searched.")

class AgenticPDFSearchTool(BaseTool):
    name: str = "agentic_pdf_search_tool"
    description: str = "Tool to search and retrieve information from PDF documents."
    args_schema: BaseModel = AgenticPDFInput
    source: str = Field(..., description="The source directory containing PDF documents.")
    
    def __init__(self, source: str, **kwargs):
        super().__init__(source=source, **kwargs)
    
    def _run(self, query: str, pdf_path: str) -> str:
        pdf_path = os.path.join(self.source, pdf_path)
        pdf_search_tool = PDFSearchTool(pdf=pdf_path)
        result = pdf_search_tool.run(query)
        return result

if __name__ == "__main__":
    tool = AgenticPDFSearchTool()
    output = tool._run(query="What is the revenue in 2020?", pdf_path="2020.pdf")
    print(output)
