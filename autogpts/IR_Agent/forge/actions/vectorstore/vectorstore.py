from ..registry import action
from ...memory.chroma_memstore import ChromaMemStore
from ...memory.chroma_vectorstore import ChromaVectorStore
from llama_index.text_splitter import SentenceSplitter
from llama_index import Document
import os


@action(
    name="ingest_document",
    description="Ingest a document into the vector database. Use this action if you need to store large files with information in your memory for later access",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file to ingest",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
async def ingest_document(
    agent,
    task_id: str,
    file_path: str,
):
    """
    Add a document to the vector databse
    """
    fulltext = agent.workspace.read_text("shared", file_path)

    splitter = SentenceSplitter(separator=".", chunk_size=128, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents([Document(text=fulltext)])

    db = ChromaVectorStore("./chroma")
    documents = [node.text for node in nodes]
    print(len(documents))
    metadatas = [{"task_id": task_id, "parent_file": file_path} for _ in range(100)]
    db.add(documents[:100], metadatas[:100])


@action(
    name="query_memory",
    description="Retrieve the most relevant information from your memory for a query and save it to a .txt file",
    parameters=[
        {
            "name": "query",
            "description": "Query used for retrieval of infomration",
            "type": "string",
            "required": True,
        },
        {
            "name": "output_file",
            "description": "Textfile to save the output to",
            "type": "string",
            "required": True,
        },
    ],
    output_type="string",
)
async def query_memory(
    agent,
    task_id: str,
    query: str,
    output_file: str,
) -> str:
    """
    Query the vector database
    """
    db = ChromaVectorStore("./chroma")
    result = db.query([query])
    # format result to save to txt file
    data = f"Query results for '{query}':\n\n"
    data += "\n\n".join(
        [f"Document {i}:\n\n{doc}" for i, doc in enumerate(result["documents"][0], 1)]
    )
    agent.workspace.write(task_id=task_id, path=output_file, data=data.encode())
    return await agent.db.create_artifact(
        task_id=task_id,
        file_name=output_file.split("/")[-1],
        relative_path=output_file,
        agent_created=True,
    )
