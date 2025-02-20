
import pandas as pd
my_df = pd.read_csv("feature_selection_sample_data.csv")


# Regreesion Template - output is number

from sklearn.feature_selection import SelectKBest, f_regression

X = my_df.drop(["output"], axis=1)
y = my_df["output"]

"""f_regression asseses the relationship between each input variable and the output
providing us F score and P value with both essentially tell us how confident we can be that
there is a true and robust relationship between the input and the output"""

feature_selecter = SelectKBest(f_regression, k="all")  
fit = feature_selecter.fit(X,y)

fit.pvalues_
"""array([6.41321253e-14, 3.11971032e-14, 3.28616228e-01, 5.11901492e-01])
Lower P value is better, there is more confident that relationship is robust."""


fit.scores_
""" array([96.13254595, 99.97291167,  0.97062608,  0.43552725]) 
a higher value indicates a stronger relationship."""

p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
inpu_variable_names = pd.DataFrame(X.columns)
summary_state = pd.concat([inpu_variable_names, p_values, scores], axis = 1)
summary_state.columns = ["input_variable", "p_value", "f_score"]
summary_state.sort_values(by = "p_value", inplace = True)

"""
  input_variable       p-value    f_score
0         input1  6.413213e-14  96.132546
1         input2  3.119710e-14  99.972912
2         input3  3.286162e-01   0.970626
3         input4  5.119015e-01   0.435527
"""

p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_state.loc[ (summary_state["f_score"] >= score_threshold) &
                                        (summary_state["p_value"] <= p_value_threshold) ]
"""
  input_variable       p_value    f_score
1         input2  3.119710e-14  99.972912
0         input1  6.413213e-14  96.132546
"""
selected_variables = selected_variables["input_variable"].tolist()
X_new = X[selected_variables]


# classification Template 

from sklearn.feature_selection import SelectKBest, chi2

X = my_df.drop(["output"], axis=1)
y = my_df["output"]

feature_selecter = SelectKBest(chi2, k="all")  
fit = feature_selecter.fit(X,y)


p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
inpu_variable_names = pd.DataFrame(X.columns)
summary_state = pd.concat([inpu_variable_names, p_values, scores], axis = 1)
summary_state.columns = ["input_variable", "p_value", "chi2_score"]
summary_state.sort_values(by = "p_value", inplace = True)

p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_state.loc[ (summary_state["chi2_score"] >= score_threshold) &
                                        (summary_state["p_value"] <= p_value_threshold) ]

selected_variables = selected_variables["input_variable"].tolist()
X_new = X[selected_variables]

















