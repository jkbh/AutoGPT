import json
from pprint import pformat

from forge.actions import ActionRegister
from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    PromptEngine,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,
    chat_completion_request,
)
from forge.sdk.errors import NotFoundError

LOG = ForgeLogger(__name__)

BEST_PRACTICES = [
    "Do not use the same ability two times in a row."
    "Prefer querying your memory to retrieve source before answering.",
    "When a proposed ability outputs an error, address that error in your next proposed ability.",
    "Speak about what you are doing in the current step.",
]
RESOURCES = [
    "A vector database representing your memory. You can ingest documents into it and query it for relevant information."
]

class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code, so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally, based on the profile selected, the agent could be configured to use a
    different llm. The possibilities are endless and the profile can be selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to accumulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensing short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agent's decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)
        self.abilities = ActionRegister(self)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """

        task = await self.db.get_task(task_id)

        prompt_engine = PromptEngine("gpt-3.5-turbo")

        # Uncomment to use full chat history in the openai messages format
        # Get chat history
        # try:
        # messages = await self.db.get_chat_history(task_id)
        # except NotFoundError:
        # LOG.info("No chat history found, creating system format")
        # system_prompt = prompt_engine.load_prompt("system-format")
        # messages = [{"role": "system", "content": system_prompt}]
        # await self.db.add_chat_message(task_id, "system", system_prompt)

        system_prompt = prompt_engine.load_prompt("system-format")
        messages = [{"role": "system", "content": system_prompt}]

        try:
            actions = await self.db.get_action_history(task_id)
        except NotFoundError:
            actions = []

        try:
            past_steps, _ = await self.db.list_steps(task_id)
        except NotFoundError:
            past_steps = []

        # actions_outputs = [
        #     (action, step.additional_output["ability_output"])
        #     for action, step in zip(actions, past_steps)
        #     if step.additional_output
        # ]

        # actions_outputs_fmt = [
        #     f"Action {i}: {action}\nOutput {i}:\n{result}"
        #     for i, (action, result) in enumerate(actions_outputs)
        # ]

        previous_actions = []
        for i, step in enumerate(past_steps, 1):
            if step.additional_output:
                error = step.additional_output["ability"].get("error", None)
                output = step.additional_output["ability"].get("output", None)
                if error:
                    previous_actions.append(
                        f"{i}. Action: {step.input}\nError: {error}"
                    )
                elif output:
                    previous_actions.append(
                        f"{i}. Action: {step.input}\nOutput: {output}"
                    )

        task_prompt = prompt_engine.load_prompt(
            "task-step",
            task=task.input,
            abilities=self.abilities.list_abilities_for_prompt(),
            best_practices=BEST_PRACTICES,
            resources=RESOURCES,
            previous_actions=previous_actions,
        )

        messages.append({"role": "user", "content": task_prompt})
        # await self.db.add_chat_message(task_id, "user", task_prompt)
        LOG.info(f"Task prompt: {pformat(task_prompt)}")

        try:
            chat_response = await chat_completion_request(
                messages=messages, model="gpt-3.5-turbo"
            )

            response_content = chat_response.choices[0].message.content
            # await self.db.add_chat_message(task_id, "assistant", response_content)
            answer = json.loads(response_content)

        except json.JSONDecodeError:
            LOG.error(f"Unable to decode chat response: {response_content}")
        except Exception as e:
            LOG.error(f"Exception: {e}")

        ## Parse LLM plan
        # plan = answer["thoughts"]["plan"].split("\n")
        # plan = [task.strip("- ") for task in plan]

        LOG.info(f"Answer: {pformat(answer)}")
        thoughts = answer["thoughts"]
        ability = answer["ability"]

        ## Create step in DB
        step = await self.db.create_step(
            task_id,
            input=step_request,
            is_last=ability["name"] == "finish",
        )

        additional_output = {
            "ability": {
                "proposed": ability,
            }
        }

        ## Execute ability
        try:
            ability_output = await self.abilities.run_action(
                task_id=task_id, action_name=ability["name"], **ability["args"]
            )
            additional_output["ability"]["output"] = str(ability_output)
        except Exception as e:
            LOG.error(f"Error trying to execute ability {ability}: {e}")
            additional_output["ability"]["error"] = str(e)

        await self.db.create_action(task_id, ability["name"], ability["args"])

        output = f'{thoughts["speak"]}'

        step = await self.db.update_step(
            task_id,
            step.step_id,
            output=output,
            additional_output=additional_output,
            status="completed",
        )

        # self.workspace.write(
        #     task_id=task_id, path="step_output.txt", data=output.encode()
        # )

        # await self.db.create_artifact(
        #     task_id=task_id,
        #     step_id=step.step_id,
        #     file_name="step_output.txt",
        #     relative_path="",
        #     agent_created=True,
        # )

        return step
