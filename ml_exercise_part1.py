import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df["MedHouseValue"] = pd.Series(california.target)

sns.set_style('whitegrid')

grid = sns.pairplot(
    data=california_df,
    vars=california_df.columns[0:4])  # only did first 4 attributes bc my computer couldn't handle more

plt.show()

# part 2 of hw in seperate file
