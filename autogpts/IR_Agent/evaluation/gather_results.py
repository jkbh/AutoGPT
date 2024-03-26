from os import path, listdir
import pandas as pd


def main():
    eval_dir = path.dirname(path.realpath(__file__))
    results_dir = path.join(eval_dir, "results")

    distances = None

    for filename in listdir(results_dir):
        if "gathered" in filename:
            continue
        df = pd.read_csv(path.join(results_dir, filename))
        if df["distances"].size < 9:
            continue
        if distances is None:
            distances = {filename: df["distances"].to_list()}
        else:
            distances[filename] = df["distances"].to_list()

    dataframe = pd.DataFrame(distances)
    dataframe.to_csv(path.join(results_dir, "gathered.csv"))


if __name__ == "__main__":
    main()
