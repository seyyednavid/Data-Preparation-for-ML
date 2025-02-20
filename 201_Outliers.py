

import pandas as pd

my_df = pd.DataFrame({"input1":[15,41,44,47,50,53,56,59,99],
                      "input2":[29,41,44,47,50,53,56,59,66]})

my_df.plot(kind="box", vert = False )

outlier_columns = ["input1", "input2"]


# Boxplot approach

for column in outlier_columns:
    lower_quartile = my_df[column].quantile(0.25)
    upper_quartile = my_df[column].quantile(0.75)
    iqr = upper_quartile -  lower_quartile
    iqr_extended = iqr * 1.5
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    # Get index as I need to remove the whole row
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detectedin column {column}")
    
    # Remove the row of outliers from dataframe
    my_df.drop(outliers, inplace = True)
    
"""
2 outliers detectedin column input1
0 outliers detectedin column input2
"""
    
    
# Standard Deviation approach

my_df = pd.DataFrame({"input1":[15,41,44,47,50,53,56,59,99],
                      "input2":[29,41,44,47,50,53,56,59,66]})

outlier_columns = ["input1", "input2"]

for column in outlier_columns:
    mean = my_df[column].mean()
    std_dev = my_df[column].std()
   
    min_border = mean - std_dev * 3
    max_border = mean + std_dev * 3
    
    # Get index as I need to remove the whole row
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detectedin column {column}")
    
    # Remove the row of outliers from dataframe
    my_df.drop(outliers, inplace = True)

"""
0 outliers detectedin column input1
0 outliers detectedin column input2
"""















    