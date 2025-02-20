
import pandas as pd

my_df = pd.DataFrame({"Height": [1.98,1.77,1.76,1.80,1.64],
                      "Weight": [99,81,70,86,82]})

# Standardization approach (based on mean and std) => (x - mean) / std
from sklearn.preprocessing import StandardScaler

scale_standard = StandardScaler()
scale_standard.fit_transform(my_df)
my_df_standardised = pd.DataFrame(scale_standard.fit_transform(my_df), columns = my_df.columns )



# Normalization approach (based on min and max) => (X - min) / (max - min)
from sklearn.preprocessing import MinMaxScaler

scale_norm = MinMaxScaler()
scale_norm.fit_transform(my_df)
my_df_normalised = pd.DataFrame(scale_norm.fit_transform(my_df), columns = my_df.columns)