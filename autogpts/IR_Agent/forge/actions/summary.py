from forge.sdk.prompting import PromptEngine
from .registry import action
from ..llm import chat_completion_request


# @action(
#     name="summarize",
#     description="Summarize a text file",
#     parameters=[
#         {
#             "name": "file",
#             "description": "Text file that should be summarized",
#             "type": "string",
#             "required": True,
#         }
#     ],
#     output_type="string",
# )
async def summarize(agent, task_id, file) -> str:
    """
    Summarize a document
    """
    prompt_engine = PromptEngine("gpt-3.5-turbo")

    text: bytes = agent.workspace.read(task_id=task_id, path=file)
    prompt = prompt_engine.load_prompt("summarize", data=text.decode())
    print(prompt)

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await chat_completion_request(
            messages=messages, model="gpt-3.5-turbo"
        )
    except Exception as e:
        print(e)

    response_content = response["choices"][0].message.content

    return response_content
