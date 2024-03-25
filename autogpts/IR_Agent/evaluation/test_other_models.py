from os import path
from openai import OpenAI
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import argparse
import logging
import cachetools


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger = logging.getLogger(__file__)
logger.addHandler(handler)
logger.setLevel(level=logging.INFO)

parser = argparse.ArgumentParser(description="Evaluate Q&A quality of different tools")
parser.add_argument(
    "source",
    choices=["gpt3", "gpt4", "agent", "perplexity"],
    help="The source to be evaluated",
)
args = parser.parse_args()

match args.source:
    case "gpt3":
        MODEL = "gpt-3.5-turbo"
    case "gpt4":
        MODEL = "gpt-4-turbo-preview"
    case s:
        MODEL = s

TEST = True

load_dotenv()
# library_dir = path.dirname(path.realpath(__file__))
eval_dir = path.dirname(path.realpath(__file__))
testdata_path = path.join(eval_dir, "test_data.txt")

questions = []
answers = []
with open(testdata_path) as file:
    for line in file:
        questions.append(line.strip())
        answers.append(next(file).strip())
        _ = next(file)  # skip empty line
        if TEST:
            break


def get_answer(question, index):
    if "gpt" in MODEL:
        answer_path = path.join(eval_dir, "answers", f"{index}_{MODEL}.txt")
        if path.exists(answer_path):
            logger.info(f"Using cached answer for question {index}")
            with open(answer_path) as file:
                answer = "\n".join(file.readlines())
        else:
            logger.info(f"Calling API for question {index}")
            client = OpenAI()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": f"{question}"},
                ],
            )
            with open(answer_path, "+x") as file:
                answer = response.choices[0].message.content
                file.write(answer)
        return answer

    model_answers_path = path.join(eval_dir, f"{MODEL}_answers.txt")
    with open(model_answers_path) as file:
        for i, line in enumerate(file):
            if i == index:
                return line


def get_embeddings(documents):
    model = "text-embedding-3-small"
    client = OpenAI()
    documents = [doc.replace("\n", " ") for doc in documents]
    response = client.embeddings.create(input=documents, model=model)
    return [item.embedding for item in response.data]


def get_distances(answers, model_answers):
    "Calculates cosine distances from embeddings"
    distances = []
    for gold, model in zip(answers, model_answers):
        distance = cosine(gold, model)
        distances.append(distance)
    return distances


model_answers = []
for i, question in enumerate(questions):
    logger.info(f"Generating answer for question {i}")
    logger.debug(f"Q: {question}")
    answer = get_answer(question, i)
    logger.debug(f"A: {answer}\n")
    model_answers.append(answer)


logger.info("Generating embeddings with OpenAI API")
answer_embeddings = get_embeddings(answers)
model_answer_embeddings = get_embeddings(model_answers)

results = pd.DataFrame(
    {
        "questions": questions,
        "answers": answers,
        "model_answers": model_answers,
        "distances": get_distances(answer_embeddings, model_answer_embeddings),
    }
)

timestamp = datetime.strftime(datetime.now(), "%m_%d_%y_%H:%M:%S")
output_path = path.join(eval_dir, "results", f"{MODEL}_{timestamp}.csv")
results.to_csv(output_path)
logger.info(f"Results written to {output_path}")
