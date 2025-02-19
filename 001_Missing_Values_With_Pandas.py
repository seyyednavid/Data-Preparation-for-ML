

import pandas as pd
import numpy as np

my_df = pd.DataFrame({"A":[1,2,4,np.nan,5,np.nan,7],
                      "B":[4,np.nan,7,np.nan,1,np.nan,2]})

# Finding Missing Values with Pandas
my_df.isna()
my_df.isna().sum()
my_df.dropna(how=any)
my_df.dropna(how=all)
my_df.fillna(value=100)

"""Using Scikit learn instead of PandasUsing scikit-learn
approch allows us to link together all the data preparation
techniques into one single pipeline """

# Simple Imputer - mean, median
# KNN Imputer