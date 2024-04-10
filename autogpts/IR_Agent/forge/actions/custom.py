from .registry import action
from ..llm import chat_completion_request, create_embedding_request
from ..sdk.prompting import PromptEngine
from ..memory.chroma_vectorstore import ChromaVectorStore


async def create_artifact(agent, task_id, file):
    return await agent.db.create_artifact(
        task_id=task_id,
        file_name=file.split("/")[-1],
        relative_path=file,
        agent_created=True,
    )


@action(
    name="retrieve_context_from_memory",
    description="Retrieve the most relevant information for a query and save it to a .txt file",
    parameters=[
        {
            "name": "query",
            "description": "The query used for retrieval",
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
    output_type="file",
)
async def retrieve_context_from_memory(agent, task_id, query: str, output_file: str):
    db = ChromaVectorStore("./chroma")
    result = db.query([query])

    data = "\n\n".join(
        [f"Document {i}:\n\n{doc}" for i, doc in enumerate(result["documents"][0], 1)]
    )
    agent.workspace.write(task_id=task_id, path=output_file, data=data.encode())
    await create_artifact(agent, task_id, output_file)
    return f"file: {output_file}"


@action(
    name="answer_with_context",
    description="Answer a prompt using content from a textfile as context",
    parameters=[
        {
            "name": "prompt",
            "description": "The prompt to generate an answer for",
            "type": "str",
            "required": True,
        },
        {
            "name": "context_file",
            "description": "The file that contains the context",
            "type": "str",
            "required": True,
        },
        {
            "name": "output_file",
            "description": "The file that contains the answer",
            "type": "str",
            "required": True,
        },
    ],
    output_type="file",
)
async def answer_with_context(
    agent, task_id, prompt: str, context_file: str, output_file: str
):
    model = "gpt-3.5-turbo"
    context = agent.workspace.read_text(task_id, context_file)

    engine = PromptEngine(model)
    prompt_with_context = engine.load_prompt(
        "augmented-generation", task=prompt, context=context
    )

    messages = [
        {
            "role": "system",
            "content": "You are a grounded and factual prompt answer assisstant. You only answer based on the provided context.",
        },
        {"role": "user", "content": prompt_with_context},
    ]

    response = await chat_completion_request(model=model, messages=messages)
    answer = response.choices[0].message.content

    agent.workspace.write(task_id=task_id, path="output.txt", data=answer.encode())
    await create_artifact(agent, task_id, "output.txt")
    return "file: output.txt"


async def generate_backlinks(agent, task_id: str, input_file: str, output_file: str):
    pass
