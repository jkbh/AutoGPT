from ..llm import chat_completion_request
from .registry import action
from ..sdk.prompting import PromptEngine


# @action(
#     name="generate_with_context",
#     description="Answer a prompt using content from a textfile as context",
#     parameters=[
#         {
#             "name": "prompt",
#             "description": "The prompt to generate an answer for",
#             "type": "str",
#             "required": True,
#         },
#         {
#             "name": "context_file",
#             "description": "The file that contains the context",
#             "type": "str",
#             "required": True,
#         },
#         {
#             "name": "out_file",
#             "description": "The file will that contains the answer",
#             "type": "str",
#             "required": True,
#         },
#     ],
#     output_type="None",
# )
# async def generate_with_context(
#     agent, task_id: str, prompt: str, context_file: str, out_file: str
# ):
#     """Generate answer to prompt based on given context"""
#     model = "gpt-3.5-turbo"
#     context = agent.workspace.read_text(task_id, context_file)

#     engine = PromptEngine(model)
#     rag_prompt = engine.load_prompt(
#         "augmented-generation", task=prompt, context=context
#     )

#     messages = [
#         {
#             "role": "system",
#             "content": "Your answer to prompts only relies on the provided context.",
#         },
#         {"role": "user", "content": rag_prompt},
#     ]

#     response = await chat_completion_request(model=model, messages=messages)
#     answer = response.choices[0].message.content

#     agent.workspace.write(task_id=task_id, path=out_file, data=answer.encode())
#     return await agent.db.create_artifact(
#         task_id=task_id,
#         file_name=out_file.split("/")[-1],
#         relative_path=out_file,
#         agent_created=True,
#     )
