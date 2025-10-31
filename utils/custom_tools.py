from crewai.tools import BaseTool
from crewai_tools import PDFSearchTool
from pydantic import BaseModel, Field
import os
import polars as pl
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class AgenticPDFInput(BaseModel):
    query: str = Field(
        ..., description="The search query to find information in the PDF."
    )
    pdf_path: str = Field(
        ..., description="The path to the PDF document to be searched."
    )


class AgenticPDFSearchTool(BaseTool):
    name: str = "agentic_pdf_search_tool"
    description: str = "Tool to search and retrieve information from PDF documents."
    args_schema: BaseModel = AgenticPDFInput
    source: str = Field(
        ..., description="The source directory containing PDF documents."
    )

    def __init__(self, source: str, **kwargs):
        super().__init__(source=source, **kwargs)

    def _run(self, query: str, pdf_path: str) -> str:
        if not self.source.replace("/", "") == pdf_path.split("/")[0]:
            pdf_path = os.path.join(self.source, pdf_path)
        pdf_search_tool = PDFSearchTool(pdf=pdf_path)
        result = pdf_search_tool.run(query)
        return result


class AgenticExcelInput(BaseModel):
    query: str = Field(
        ..., description="The search query to find information in the Excel."
    )
    excel_path: str = Field(
        ..., description="The path to the Excel document to be searched."
    )
    file_type: str = Field(
        ..., description="The type of the Excel file, e.g., xlsx, xls, csv, or etc."
    )


class AgenticExcelSearchTool(BaseTool):
    name: str = "agentic_excel_search_tool"
    description: str = "Search and retrieve information from Excel workbooks."
    args_schema: BaseModel = AgenticExcelInput
    source: str = Field(..., description="Directory containing Excel documents.")

    def create_embedding(self, docs: list[Document], persist_directory: str) -> Chroma:
        embeddings = OpenAIEmbeddings()
        return Chroma.from_documents(
            docs, embeddings, persist_directory=persist_directory
        )

    def _run(self, query: str, excel_path: str, file_type: str) -> str:
        if not self.source.replace("/", "") == excel_path.split("/")[0]:
            excel_path = os.path.join(self.source, excel_path)
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"{excel_path} not found")
        
        persist_directory = f"./excel_index/{excel_path.split('/')[-1].replace(f'.{file_type}', '')}"
        print(f"Persist directory has been created: {persist_directory}")
        
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
        else:
            sheets = pl.read_csv(excel_path) if file_type == "csv" else pl.read_excel(excel_path, sheet_id=0)
            docs = []
            for sheet_name, sheet_df in sheets.items():
                excel_df = sheet_df.to_pandas()
                i = 0
                chunk_size = 50
                overlap = 15
                while i < len(excel_df):
                    chunk = excel_df.iloc[i : i + chunk_size].to_markdown()
                    docs.append(
                        Document(page_content=chunk, metadata={"sheet_name": sheet_name})
                    )
                    i += chunk_size - overlap
            vector_store = self.create_embedding(docs, persist_directory)
        results = vector_store.similarity_search(query, k=3)

        return {
            "matches": [
                {"sheet": r.metadata["sheet_name"], "content": r.page_content}
                for r in results
            ]
        }


if __name__ == "__main__":
    tool = AgenticExcelSearchTool(source="knowledge")
    output = tool._run(query="What is the revenue in 2020?", excel_path="knowledge/rexit/Rexit Valuation.xlsm", file_type="xlsm")
    print(output)
