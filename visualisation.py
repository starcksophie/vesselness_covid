import matplotlib.pyplot as plt
import json

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def visualize_unique(data):
    plt.figure(figsize = (15, 10))
    plt.subplot(211)
    plt.hist(data["mean_all_vessel"])
    plt.title("Mean value for each vessel (connected component)")
    plt.subplot(212)
    plt.hist(data["max_all_vessel"])
    plt.title("Max value for each vessel (connected component)")
    plt.show()


data = load_json("now.json")
visualize(data)
