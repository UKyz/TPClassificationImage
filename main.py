from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np

def main():
    a = DataManager()
    a.loadData()
    b = NeuralNetwork()
    b.createModel()
    pass

if __name__ == "__main__":
    main()
