from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np

def main():
    a = DataManager()
    a.loadData()
    b = NeuralNetwork()
    b.createModel()

    b.train(a.train_data, a.train_labels, epochs=6)
    res = b.evaluate(a.eval_data, a.eval_labels)
    print("Accuracy : {}".format(res))
    # b.saveModel()

    pass

if __name__ == "__main__":
    main()
