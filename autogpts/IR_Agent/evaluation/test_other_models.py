from os import path, listdir
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import argparse
import logging


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger = logging.getLogger(__file__)
logger.addHandler(handler)
logger.setLevel(level=logging.INFO)


def main():
    arg_model_map = {
        "gpt3": "gpt-3.5-turbo",
        "gpt4": "gpt-4-turbo-preview",
        "agent": "agent",
        "perplexity": "perplexity",
    }

    parser = argparse.ArgumentParser(
        description="Evaluate Q&A quality of different tools"
    )
    parser.add_argument(
        "--sources",
        action="append",
        choices=["gpt3", "gpt4", "agent", "perplexity"],
        help="The sources to be evaluated, if none evaluates all sources",
    )
    parser.add_argument("--test", help="Only use a single question to test the script")
    parser.add_argument("--regen", action="store_true")
    args = parser.parse_args()

    use_cache = False if args.regen else True

    match args.sources:
        case None:
            models = arg_model_map.values()
        case source_list:
            models = [arg_model_map[source] for source in source_list]

    is_test_run = True if args.test else False

    load_dotenv()
    eval_dir = path.dirname(path.realpath(__file__))
    testdata_path = path.join(eval_dir, "test_data.txt")

    questions = []
    answers = []
    with open(testdata_path) as file:
        for line in file:
            questions.append(line.strip())
            answers.append(next(file).strip())
            _ = next(file)  # skip empty line
            if is_test_run:
                break

    answer_embeddings = get_embeddings(answers)

    for model in models:
        logger.info(f"Generating results for {model}")
        logger.info("Generating embeddings with OpenAI API")
        model_answers = []
        for i, question in enumerate(questions):
            logger.info(f"Generating answer for question {i}")
            logger.debug(f"Q: {question}")

            answer = get_answer(question, i, model, use_cache)

            logger.debug(f"A: {answer}\n")

            model_answers.append(answer)

        model_answer_embeddings = get_embeddings(model_answers)

        results = pd.DataFrame(
            {
                "question": questions,
                "answer": answers,
                "model_answer": model_answers,
                "cosine_sims": get_similarities(
                    answer_embeddings, model_answer_embeddings
                ),
            }
        )

        timestamp = datetime.strftime(datetime.now(), "%m_%d_%y_%H:%M:%S")
        output_path = path.join(eval_dir, "results", f"{model}_{timestamp}.csv")
        results.to_csv(output_path)
        logger.info(f"Results written to {output_path}")

    gather_results()


def get_answer(question, index, model, use_cache):
    eval_dir = path.dirname(path.realpath(__file__))
    if "gpt" in model:
        cached_answer_path = path.join(eval_dir, "answers", f"{index}_{model}.txt")
        if use_cache and path.exists(cached_answer_path):
            logger.info(f"Using cached answer for question {index}")
            with open(cached_answer_path) as file:
                answer = "\n".join(file.readlines())
        else:
            logger.info(f"Calling API for question {index}")
            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": f"{question}"},
                ],
            )
            with open(cached_answer_path, "+x") as file:
                answer = response.choices[0].message.content
                file.write(answer)
        return answer

    # Non GPT
    model_answers_path = path.join(eval_dir, "answers", f"{model}_answers.txt")
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


def get_similarities(answers, model_answers):
    "Calculates cosine similarities from embeddings"
    similarities = []
    for gold, model in zip(answers, model_answers):
        cosine_sim = cosine_similarity([gold], [model])
        similarities.append(cosine_sim[0, 0])
    return similarities


def gather_results():
    eval_dir = path.dirname(path.realpath(__file__))
    results_dir = path.join(eval_dir, "results")

    cosine_sims = None

    for filename in listdir(results_dir):
        if "gathered" in filename:
            continue
        df = pd.read_csv(path.join(results_dir, filename))
        if df["cosine_sims"].size < 9:
            continue
        if cosine_sims is None:
            cosine_sims = {filename: df["cosine_sims"].to_list()}
        else:
            cosine_sims[filename] = df["cosine_sims"].to_list()

    dataframe = pd.DataFrame(cosine_sims)
    dataframe.to_csv(path.join(results_dir, "gathered.csv"))


if __name__ == "__main__":
    main()
