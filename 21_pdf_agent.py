from agno.agent import Agent
from agno.tools.tavily import TavilyTools
from agno.models.groq import Groq
from agno.storage.sqlite import SqliteStorage
from agno.playground import Playground, serve_playground_app

from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb

from dotenv import load_dotenv
load_dotenv()

vector_db = ChromaDb(collection="pdf_agent", path="tmp/chromadb")

knowledge = PDFKnowledgeBase(
path="GlobalEVOutlook2025.pdf",
vector_db=vector_db,
reader=PDFReader(chunk=True),
)
knowledge.load()

db = SqliteStorage(table_name="agent_session", db_file="tmp/agent.db")

agent = Agent(
    name="Conversor de Temperatura",
    model=Groq(id="llama-3.3-70b-versatile"),
    storage=db,
    knowledge=knowledge,
    add_history_to_messages=True,
    num_history_runs=3
)

app = Playground(agents=[
    agent
]).get_app()

if __name__ == "__main__":
    serve_playground_app("21_pdf_agent:app", reload=True)