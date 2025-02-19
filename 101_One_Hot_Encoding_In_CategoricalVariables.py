
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

X = pd.DataFrame({"input1": [1,2,3,4,5],
                  "input2": ["A","A","B","B","C"],
                  "input3": ["X","X","X","Y","Y"],})


categorical_val = ["input2", "input3"]

# If sparse_output=False, the output is a dense NumPy array
one_hot_encoder = OneHotEncoder(sparse_output= False)

encoder_vars_arrays = one_hot_encoder.fit_transform(X[categorical_val])
""""
array([[1., 0., 0., 1., 0.],
       [1., 0., 0., 1., 0.],
       [0., 1., 0., 1., 0.],
       [0., 1., 0., 0., 1.],
       [0., 0., 1., 0., 1.]])
"""

encoder_features_name = one_hot_encoder.get_feature_names_out(categorical_val)
"""
array(['input2_A', 'input2_B', 'input2_C', 'input3_X', 'input3_Y'],
      dtype=object)
"""

encoder_vars_df = pd.DataFrame(encoder_vars_arrays, columns=encoder_features_name)
"""
   input2_A  input2_B  input2_C  input3_X  input3_Y
0       1.0       0.0       0.0       1.0       0.0
1       1.0       0.0       0.0       1.0       0.0
2       0.0       1.0       0.0       1.0       0.0
3       0.0       1.0       0.0       0.0       1.0
4       0.0       0.0       1.0       0.0       1.0
"""


X_new = pd.concat([X.reset_index(drop=True),encoder_vars_df.reset_index(drop=True)], axis=1)
"""
   input1 input2 input3  input2_A  input2_B  input2_C  input3_X  input3_Y
0       1      A      X       1.0       0.0       0.0       1.0       0.0
1       2      A      X       1.0       0.0       0.0       1.0       0.0
2       3      B      X       0.0       1.0       0.0       1.0       0.0
3       4      B      Y       0.0       1.0       0.0       0.0       1.0
4       5      C      Y       0.0       0.0       1.0       0.0       1.0
"""

X_new.drop(categorical_val, axis = 1, inplace=True)
"""
   input1  input2_A  input2_B  input2_C  input3_X  input3_Y
0       1       1.0       0.0       0.0       1.0       0.0
1       2       1.0       0.0       0.0       1.0       0.0
2       3       0.0       1.0       0.0       1.0       0.0
3       4       0.0       1.0       0.0       0.0       1.0
4       5       0.0       0.0       1.0       0.0       1.0
"""
#---------------------------------------------------------------------
"""
when we are using one hot encoding depending on the type of model that
we are applying, we can fall into something called the dummy variable trap,
which is where input variables perfectly predict each other, and this violates
an assumption(of regression models like linear regression and logistic regression)
 of something called multicollinearity while this sounds like a big problem, for handling
"""
one_hot_encoder = OneHotEncoder(sparse_output= False, drop="first")
# One of the encoded columns is always removed.
"""
   input1  input2_B  input2_C  input3_Y
0       1       0.0       0.0       0.0
1       2       0.0       0.0       0.0
2       3       1.0       0.0       0.0
3       4       1.0       0.0       1.0
4       5       0.0       1.0       1.0
"""