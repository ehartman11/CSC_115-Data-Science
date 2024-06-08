import pandas as pd
from scipy import stats

# Load your dataset into a dataFrame
df = pd.DataFrame(data='')

# Calculate z-scores for each column
# z = (data point)(standard deviation) - (mean)
z_score = stats.zscore(df)

# set threshold for identifying outliers
threshold = 3

# Identify outliers based on z-score
outliers = df[(z_score > threshold).any(axis=1)]

# Remove outliers from the dataset
df_cleaned = df[(z_score <= threshold).all(axis=1)]

# Another method is the Interquartile range method
# Calculate Q1, Q3, and IQR for each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Set threshold for identifying outliers
threshold = 1.5

# Identify outliers based on IQR
outlier = df[((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)]

# Remove outliers from the dataset
df_cleaned = df[~((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)]
