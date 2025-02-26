import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NHANES datasets
nhanes_2011_2012 = pd.read_csv('processed_data/2011-2012_NHANES.csv')
nhanes_2013_2014 = pd.read_csv('processed_data/2013-2014_NHANES.csv')

# Display the first few rows of each dataset
print(nhanes_2011_2012.head())
print(nhanes_2013_2014.head())

# Basic analysis: Summary statistics
summary_2011_2012 = nhanes_2011_2012.describe()
summary_2013_2014 = nhanes_2013_2014.describe()

print("Summary Statistics for 2011-2012:")
print(summary_2011_2012)
print("\nSummary Statistics for 2013-2014:")
print(summary_2013_2014)

# Visualization: Age distribution in 2011-2012
plt.figure(figsize=(10, 6))
sns.histplot(nhanes_2011_2012['RIDAGEYR'], bins=30, kde=True)
plt.title('Age Distribution (2011-2012)')
plt.xlabel('Age (Years)')
plt.ylabel('Frequency')
plt.show()

# Visualization: Gender distribution in 2013-2014
plt.figure(figsize=(8, 5))
sns.countplot(x='RIAGENDR', data=nhanes_2013_2014)
plt.title('Gender Distribution (2013-2014)')
plt.xlabel('Gender (1=Male, 2=Female)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()

# Visualization: Gender distribution in 2011-2012
plt.figure(figsize=(8, 5))
sns.countplot(x='RIAGENDR', data=nhanes_2011_2012)
plt.title('Gender Distribution (2011-2012)')
plt.xlabel('Gender (1=Male, 2=Female)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
plt.show()

# Visualization: Age distribution in 2013-2014
plt.figure(figsize=(10, 6))
sns.histplot(nhanes_2013_2014['RIDAGEYR'], bins=30, kde=True)
plt.title('Age Distribution (2013-2014)')
plt.xlabel('Age (Years)')
plt.ylabel('Frequency')
plt.show()