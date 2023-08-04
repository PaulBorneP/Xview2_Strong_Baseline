from collections import defaultdict
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import tqdm
import pandas as pd

root = '/local_storage/datasets/sgerard/xview2/'


def event_distribution(root):
    d = {}
    for part in ["train", "tier3", "test"]:
        d[part] = defaultdict(lambda: 0)
        for filename in glob.glob(root + part + "/images/*.png"):
            situation = filename.split("/")[-1].split("_")[0]
            d[part][situation] += 1

    d["train_tier3"] = defaultdict(lambda: 0)
    for k in list(d["train"].keys()) + list(d["tier3"].keys()):
        d["train_tier3"][k] = d["train"][k] + d["tier3"][k]

    d["train_tier3_test"] = defaultdict(lambda: 0)
    for k in list(d["train_tier3"].keys()) + list(d["test"].keys()) :
        d["train_tier3_test"][k] = d["train_tier3"][k] + \
            d["test"][k] 
    # print the total number of events
    print(d["train_tier3_test"].values())
    print("Total number of events: ", sum(d["train_tier3_test"].values()))

    for part in ["train_tier3", "test",  "train_tier3_test"]:
        print(part, ":")
        for k in sorted(d[part].keys()):
            print(f"{k}: {d[part][k]}")
            plt.figure(figsize=(20, 13))
            plt.title = part
            sns.barplot(x=list(d[part].keys()), y=list(d[part].values()))
            plt.xticks(rotation=45)
            plt.xlabel("Event")
            plt.ylabel("Count")
            plt.savefig(f"/Midgard/home/paulbp/{part}.png")
            plt.close()

        print("\n")


def class_distribution(root):
    """for  a given dataset find the distribution of events like above. Then for each event look at the value of the pixel of each image using opencv in the class to get the distribution of the classes"""
    d = {}
    for part in ["train", "tier3", "test"]:
        d[part] = {}
        for filename in tqdm.tqdm(glob.glob(os.path.join(root + part + "/masks/*post_disaster.png"))):
            event = filename.split("/")[-1].split("_")[0]
            if event not in d[part]:
                d[part][event] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for i in range(0, 5):
                d[part][event][i] += len(np.where(cv2.imread(filename) == i)[0])

    for part in ["train", "tier3", "test"]:
        dict = []
        for k in sorted(d[part].keys()):
            dict.append({"event": k, **d[part][k]})
        df = pd.DataFrame(dict)
        # write the distribution of the classes for each event in a csv file
        df.to_csv(f"/Midgard/home/paulbp/{part}_distrib.csv")
        # plot a stacked barplot for each event and the distribution of the classes in log scale with a legend to say what color represent what class
        plt.close()


def plot_distributions():
    df1 = pd.read_csv(f"/Midgard/home/paulbp/train_distrib.csv", index_col=0)
    df2 = pd.read_csv(f"/Midgard/home/paulbp/tier3_distrib.csv", index_col=0)
    df = pd.concat((df1, df2))
    print(df)
    plt.figure(figsize=(20, 13))
    plt.bar(df["event"], df["1"], label="no damage")
    plt.bar(df["event"], df["2"], bottom=df["1"], label="minor damage")
    plt.bar(df["event"], df["3"], bottom=df["1"] +
            df["2"], label="major damage")
    plt.bar(df["event"], df["4"], bottom=df["1"] +
            df["2"] + df["3"], label="destroyed")
    plt.legend()

    plt.yscale("log")
    plt.title(f"train_tier2 : distribution of classes")
    plt.xticks(rotation=45)
    plt.xlabel("Event")
    plt.ylabel("Count")
    plt.savefig(f"/Midgard/home/paulbp/plots_distribs/train_tier3_distrib.png")
    plt.close()



if __name__ == "__main__":
    # event_distribution(root)
    class_distribution(root)
    # plot_distributions()
