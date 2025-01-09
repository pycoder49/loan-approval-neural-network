from data import Data

import pandas as pd
import matplotlib.pyplot as plt
import torch

data = pd.read_csv("loan_approval_dataset.csv")

# prepping data
data_obj = Data(data)
data_obj.clean_data()
data_obj.transform()
data = data_obj.get_data()

print(data)

# normalizing data



