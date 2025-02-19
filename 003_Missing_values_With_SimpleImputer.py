
import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer


my_df = pd.DataFrame({"A":[1,4,7,10,13],
                      "B":[3,6,9,np.nan,15],
                      "C":[2,5,np.nan,11,np.nan]})

imputer = SimpleImputer() # default is mean of columns

imputer.fit(my_df)
imputer.transform(my_df)

my_df1 = imputer.transform(my_df)


# We can do fit and transform both at once
imputer.fit_transform(my_df)
""" Only use fit transform on training data.
if we wanna apply the same logic for test data or
to new data fit and transform seperately"""