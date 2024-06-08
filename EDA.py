import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/siglimumuni/Datasets/master/customer-data.csv')
print(df.shape)
print(df.columns)
print(df.info())
print(df.isna().sum())
print(df.groupby('income')['credit_score'].mean())


# create a function to input missing value based on mean credit score for each income group
def input_credit_score(income_classes):
    """
    This function takes a list of income groups and inputs the missing values of each
    based on the mead credit score for each group
    :param income_class: list
    :return: None
    """
    # Iterate through each income group
    for income_class in income_classes:
        # Create a subset of dataframe to use as filter
        mask = df['income'] == income_class

        # Calculate the mean for the income group
        mean = df[df['income'] == income_class]['credit_score'].mean()

        # fill the missing values with mean of credit score for group
        df.loc[mask, 'credit_score'] = df.loc[mask, 'credit_score'].fillna(mean)


# Apply the function to the dataframe
income_groups = ['poverty', 'upper class', 'middle class', 'working class']
input_credit_score(income_groups)

# Check for missing values
print(df.isnull().sum())

# The mean for the different groups in the driving_experience column do not vary to widely
# So we can simply input the null values using the column mean

# Check the mean annual mileage for the different driving experience groups
print(df.groupby('driving_experience')['annual_mileage'].mean())

# Calculate mean for annual milage column
mean_mileage = df['annual_mileage'].mean()

# fill in null values using the column mean
df['annual_mileage'].fillna(mean_mileage, inplace=True)

# check for null values
print(df.isna().sum())

# Both the id and postal_code columns are not relevant for the analysis,
# So we can get rid of these using drop.
# axis=1 will indicate columns, inplace=True will make the change permanent
# delete the id and postal_code columns
df.drop(['id', 'postal_code'], axis=1, inplace=True)


# Univariate analysis is the simplest for of analyzing data.
# It deals with analyzing data within a single column or variable.
# Categorical unordered: No order or ranking, and is categorical rather than numerical

# Check the count for each category in the 'gender' column
print(df['gender'].value_counts())
# Now create a visualization
sns.countplot(data=df, x='gender')
plt.title('Number of Clients per gender')
plt.ylabel('Number of Clients')
plt.show()

# Categorical ordered: Has a natural rank and progression
# Define plot size
plt.figure(figsize=[6, 6])

# Define column to use
data = df['income'].value_counts(normalize=True)

# Define labels
labels = ['upper class', 'middle class', 'poverty', 'working class']

# Define color palette
colors = sns.color_palette('pastel')

# Create pie chart
plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
plt.title('Proportion of Clients by Income Group')
plt.show()

# Create a countplot to visualize the count o each category in the education column
plt.figure(figsize=[8, 5])
sns.countplot(data=df, x='education', order=['university', 'high school', 'none'], color='orange')
plt.title('Number of Clients per Education Level')
plt.show()

# Numeric: usually analyzed by calculating functions like the mean, mode, max, min, std dev etc.
# summary statistics can be obtained on numerical columns by using describe()
# Return summary statistics for the 'credit_score' column
print(df['credit_score'].describe())

# Plot a histogram using the 'credit_score' column
plt.figure(figsize=[8, 5])
sns.histplot(data=df, x='credit_score', bins=40).set(title='Distribution of credit scores', ylabel='Number of clients')
plt.show()

# kernel density estimation (kde) is used to show smoothness or continuity
# plot a histogram using the 'annual milage' column
plt.figure(figsize=[8, 5])
sns.histplot(data=df, x='annual_mileage', bins=20, kde=True).set(title='Distribution of Annual Mileage', ylabel='Number of clients')
plt.show()


# Bivariate analysis: analysis using two variables or columns
# Usually to explore the relationships between variables and how they influence each other, if at all.
# Create a scatter plot to show relationship between 'annual_mileage' and 'speeding_violations'
plt.figure(figsize=[8, 5])
plt.scatter(data=df, x='annual_mileage', y='speeding_violations')
plt.title('Annual Mileage vs. Speeding Violations')
plt.ylabel('Speeding Violations')
plt.xlabel('Annual Mileage')
plt.show()

# A correlation matrix is useful for identifying the relationship between several variables
# Create a correlation matrix to show relationship between select variables
corr_matrix = df[['speeding_violations', 'DUIs', 'past_accidents']].corr()
print(corr_matrix)
# Generally, a correlation coefficient between 0.5 and 0.7 indicates variables that can be considered moderately correlated
# 0.3 to 0.5 indicates weak correlation.
# Create a heat map to visualize correlation
plt.figure(figsize=[8, 5])
sns.heatmap(corr_matrix, annot=True, cmap='Reds')
plt.title('Correlation between Selected Variables')
plt.show()

# Check the mean annual mileage per category in the outcome column
print(df.groupby('outcome')['annual_mileage'].mean())

# Box plots display a five-number summary of a set of data; the min, first quartile, median, third quartile, and max
# Plot two boxplots to compare dispersion
sns.boxplot(data=df, x='outcome', y='annual_mileage')
plt.title('Distribution of Annual Mileage per Outcome')
plt.show()

# Create histograms to compare distribution
sns.histplot(df, x='credit_score', hue='outcome', element='step', stat='density')
plt.title('Distribution of Credit Score per Outcome')
plt.show()

# Create a new 'claim rate' column
df['claim_rate'] = np.where(df['outcome'] == True, 1, 0)
print(df['claim_rate'].value_counts())

# Plot the average claim rate per age group
plt.figure(figsize=[8, 5])
df.groupby('age')['claim_rate'].mean().plot(kind='bar')
plt.title('Claim Rate by Age Group')
plt.show()

# Plot the average claim rate per vehicle year category
plt.figure(figsize=[8, 5])
df.groupby('vehicle_year')['claim_rate'].mean().plot(kind='bar')
plt.title('Claim Rate by Vehicle Year')
plt.show()

# Create an empty figure object
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot two probability graphs for education and income
for i, col in enumerate(['education', 'income']):
    sns.histplot(df, ax=axes[i], x=col, hue='outcome', stat='probability', multiple='fill', shrink=.8, alpha=0.7)
    axes[i].set(title='Claim Probability by ' + col, ylabel=' ', xlabel=' ')

# Create a pivot table for education and income with average claim rate as values
edu_income = pd.pivot_table(data=df, index='education', columns='income', values='claim_rate', aggfunc='mean')
print(edu_income)

# Create a heatmap to visualize income, education, and claim rate
plt.figure(figsize=[8, 5])
sns.heatmap(edu_income, annot=True, cmap='coolwarm', center=0.117)
plt.title('Education Level and Income Class')
plt.show()

# Create pivot table for driving experience and marital status with average claim rate as values
driv_married = pd.pivot_table(data=df, index='driving_experience', columns='married', values='claim_rate')

# Create a heatmap to visualize driving experience, marital status, and claim rate
plt.figure(figsize=[8, 5])
sns.heatmap(driv_married, annot=True, cmap='coolwarm', center=0.117)
plt.title('Driving Experience and Marital Status')
plt.show()

# Create pivot table for gender and family status with average claim rate as values
gender_children = pd.pivot_table(data=df, index='gender', columns='children', values='claim_rate')

# Create a heatmap to visualize gender, family status and claim rate
plt.figure([8, 5])
sns.heatmap(gender_children, annot=True, cmap='coolwarm', center=0.117)
plt.title('Gender and Family Status')
plt.show()
