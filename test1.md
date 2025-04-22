# Step 0 - intro
# Welcome!

Hey there! 👋 Welcome to this interactive walkthrough of a fun and realistic dataset involving **subscription fees, customer demographics, and card categories**.

I'll be your friendly data science guide today — so grab your coffee (or tea ☕), and let's dive into this step-by-step together like we're exploring it live!

# Step 1 - Loading & Exploring the Data
We're starting off by importing some essentials: `pandas` and `numpy`, and then reading in the dataset.

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("realistic_subscription_data.csv")

# Show basic info
print("Shape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# Show first few rows
print("\nHead of dataset:")
print(df.head())
```

### ✅ Summary:
- We're working with **1,000 rows** and **7 columns**
- It includes details like customer name, age, income, card category, subscription fee, and transaction dates

### 🧐 Output: Sample Rows

| Customer Name     | Date       | Month | Card Category     | Customer Age | Income | Subscription Fee |
|------------------|------------|-------|-------------------|--------------|--------|------------------|
| Allison Hill     | 2023-05-02 | 4     | Platinum Rewards  | 45           | 11097  | 106              |
| Noah Rhodes      | 2023-05-11 | 2     | Basic             | 33           | 2871   | 33               |

> **Looking at the dataset**, I noticed that the `Month` column may not match the month in the `Date` field — we'll explore that soon!

# Step 2 - Data Cleaning & Feature Engineering
We're checking for missing values, converting date strings into `datetime`, and extracting helpful features from it like year, weekday, etc.

```python
print("Missing values:\n", df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print("\nNumber of invalid dates:", df['Date'].isna().sum())

df['Year'] = df['Date'].dt.year
df['Month_Extracted'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()

print("\nSample rows after datetime processing:")
print(df[['Date', 'Year', 'Month', 'Month_Extracted', 'Weekday']].head())
```

### 🔎 Output: Date Parsing and Features

| Date       | Year | Month | Month_Extracted | Weekday  |
|------------|------|-------|------------------|----------|
| 2023-05-02 | 2023 | 4     | 5                | Tuesday  |

> 🧠 **One interesting thing that stood out**: The `Month` column doesn't always match the actual date's month. This could be a data issue or a mislabeling. Good to keep an eye on this!

# Step 3 - Distributions & Descriptive Statistics
Time to look at the spread of key numerical variables!

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic stats
print(df[['Customer Age', 'Income', 'Subscription Fee']].describe())

# Histograms
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Customer Age'], kde=True, ax=axs[0], bins=20)
axs[0].set_title('Distribution of Customer Age')

sns.histplot(df['Income'], kde=True, ax=axs[1], bins=30)
axs[1].set_title('Distribution of Income')

sns.histplot(df['Subscription Fee'], kde=True, ax=axs[2], bins=20)
axs[2].set_title('Distribution of Subscription Fee')

plt.tight_layout()
plt.show()
```

### 📈 Output: Summary Stats

- **Mean Age**: 37  
- **Mean Income**: 6671  
- **Mean Subscription Fee**: ~$60  

> 👀 **Let's pause and look at** these distributions: most incomes cluster below 10K, and subscription fees seem to vary widely, suggesting tiered services.

# Step 4 - Categorical Features
Let's explore how card types and weekdays affect subscription behavior.

```python
print("Card Categories:", df['Card Category'].unique())
print("\nCard Category Counts:\n", df['Card Category'].value_counts())

sns.countplot(data=df, x='Card Category', order=df['Card Category'].value_counts().index)
plt.title('Distribution of Card Categories')
plt.xticks(rotation=15)
plt.show()

print("\nWeekday Counts:\n", df['Weekday'].value_counts())

sns.countplot(data=df, x='Weekday', order=df['Weekday'].value_counts().index)
plt.title('Transactions by Weekday')
plt.show()
```

### 🧾 Output: Category Counts

- **Top Card**: Silver (272 users)  
- **Top Weekday**: Friday, followed closely by Sunday and Saturday  

> 💡 **One thing to watch out for here is** the balance in card types — this might affect model training if we were to predict card behavior.

# Step 5 - Outlier Detection
Let's use IQR (Interquartile Range) to find outliers in age, income, and fee.

```python
def detect_outliers_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return len(outliers), lower, upper

for col in ['Customer Age', 'Income', 'Subscription Fee']:
    count, lower, upper = detect_outliers_iqr(col)
    print(f"{col} -> Outliers: {count}, Lower Bound: {lower:.2f}, Upper Bound: {upper:.2f}")
```

### ⚠️ Output: Outliers

- **Customer Age**: 11 outliers (older folks)  
- **Income & Subscription Fee**: No extreme outliers  

> ✅ That's a relief — most of our numerical features look well-distributed!

# Step 6 - Correlation Analysis
```python
sns.heatmap(df[['Customer Age', 'Income', 'Subscription Fee']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Numerical Features')
plt.show()
```

### 🔍 Output: Correlation Heatmap

> 🔎 Not much correlation between age and income, but **subscription fee has a modest positive correlation with income**, which makes intuitive sense.

# Step 7 - Feature Importances Visualization
### Goal:
Understand how numerical features (`Income`, `Customer Age`) correlate with `Subscription Fee` using **statistics + visualizations**.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlations
correlation_matrix = df[['Customer Age', 'Income', 'Subscription Fee']].corr()
target_corr = correlation_matrix['Subscription Fee'].sort_values(ascending=False)

print("Correlation with Subscription Fee:\n", target_corr)

# Plot 1: Bar plot of correlations
plt.figure(figsize=(8, 4))
sns.barplot(x=target_corr.index, y=target_corr.values, palette="Blues_d")
plt.title("Correlation with Subscription Fee")
plt.ylim(-0.1, 1.0)
plt.ylabel("Pearson Correlation Coefficient")
plt.axhline(y=0, color='black', linestyle='--')
plt.show()

# Plot 2: Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df, x='Income', y='Subscription Fee', ax=axes[0], alpha=0.6, color='blue')
axes[0].set_title("Income vs. Subscription Fee (Corr = 0.93)")
sns.scatterplot(data=df, x='Customer Age', y='Subscription Fee', ax=axes[1], alpha=0.6, color='green')
axes[1].set_title("Age vs. Subscription Fee (Corr = 0.16)")
plt.tight_layout()
plt.show()
```

### 📊 Output & Interpretation

#### Numerical Correlation Coefficients
```
Correlation with Subscription Fee:
Subscription Fee    1.000000
Income              0.933665
Customer Age        0.159394
Name: Subscription Fee, dtype: float64
```

#### 💡 Key Insight:
Subscription Fee has a strong correlation with Income, and a weak correlation with Age — great for feature selection!
