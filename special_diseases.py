<<<<<<< HEAD
import pandas as pd
import numpy as np  


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
class Alzheimer():
    def alzheimer():
        alzheimer = pd.read_csv("XAI_Assistant_ML\data\Alzheimer.csv")
        alzheimer = alzheimer.apply(LabelEncoder().fit_transform)
        healthy = alzheimer[alzheimer["Group"] != "Demented"]
        del healthy["Group"]
        average = healthy.mean()
        return average

class Diabetes():
    def diabetes():
        diab = pd.read_csv("XAI_Assistant_ML\data\diabetes.csv")
        healthy = diab[diab["Outcome"] == 0]
        del healthy["Outcome"]
        average = healthy.mean()
        return average

class Heart():
    def heart():
        heart = pd.read_csv("XAI_Assistant_ML\data\heart_disease_data.csv")
        healthy = heart[heart["target"] == 0]
        healthy = healthy.drop(["target"],axis=1)
        average = healthy.mean()
        return average


def main():
    pass

if __name__ == '__main__':
=======
import pandas as pd
import numpy as np  


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
class Alzheimer():
    def alzheimer():
        alzheimer = pd.read_csv("XAI_Assistant_ML\data\Alzheimer.csv")
        alzheimer = alzheimer.apply(LabelEncoder().fit_transform)
        healthy = alzheimer[alzheimer["Group"] != "Demented"]
        del healthy["Group"]
        average = healthy.mean()
        return average

class Diabetes():
    def diabetes():
        diab = pd.read_csv("XAI_Assistant_ML\data\diabetes.csv")
        healthy = diab[diab["Outcome"] == 0]
        del healthy["Outcome"]
        average = healthy.mean()
        return average

class Heart():
    def heart():
        heart = pd.read_csv("XAI_Assistant_ML\data\heart_disease_data.csv")
        healthy = heart[heart["target"] == 0]
        healthy = healthy.drop(["target"],axis=1)
        average = healthy.mean()
        return average


def main():
    pass

if __name__ == '__main__':
>>>>>>> 49c7fde11ccd284ef92fcfd1b93ce1c721e992e2
    main()