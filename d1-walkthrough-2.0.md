Hey there! 👋 Welcome to this interactive walkthrough of a fun and realistic dataset involving **subscription fees, customer demographics, and card categories**. I’ll be your friendly data science guide today — so grab your coffee (or tea ☕), and let’s dive into this step-by-step together like we're exploring it live!

---

## 📂 Step 1: Loading & Exploring the Data

We’re starting off by importing some essentials: `pandas` and `numpy`, and then reading in the dataset.

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
- We're working with **1,000 rows** and **7 columns**.
- It includes details like customer name, age, income, card category, subscription fee, and transaction dates.

---

### 🧐 Output: Sample Rows

| Customer Name     | Date       | Month | Card Category     | Customer Age | Income | Subscription Fee |
|------------------|------------|-------|-------------------|--------------|--------|------------------|
| Allison Hill     | 2023-05-02 | 4     | Platinum Rewards  | 45           | 11097  | 106              |
| Noah Rhodes      | 2023-05-11 | 2     | Basic             | 33           | 2871   | 33               |

> **Looking at the dataset**, I noticed that the `Month` column may not match the month in the `Date` field — we’ll explore that soon!

---

## 🔍 Step 2: Data Cleaning & Feature Engineering

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

> 🧠 **One interesting thing that stood out**: The `Month` column doesn’t always match the actual date's month. This could be a data issue or a mislabeling. Good to keep an eye on this!

---

## 📊 Step 3: Distributions & Descriptive Statistics

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
![plo1](https://github.com/user-attachments/assets/61850f19-e39e-4eda-ac52-b9d6be81e08d)

- Mean Age: 37
- Mean Income: 6671
- Mean Subscription Fee: ~$60

> 👀 **Let’s pause and look at** these distributions: most incomes cluster below 10K, and subscription fees seem to vary widely, suggesting tiered services.

---

## 🏷️ Step 4: Categorical Features

Let’s explore how card types and weekdays affect subscription behavior.

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
![plot2](https://github.com/user-attachments/assets/c16cdf37-6e80-4bcb-ab2a-140caf67374d)


![plot2](https://github.com/user-attachments/assets/5225c894-bff5-4372-89ac-3fa972494ede)

- **Top Card:** Silver (272 users)
- **Top Weekday:** Friday, followed closely by Sunday and Saturday

> 💡 **One thing to watch out for here is** the balance in card types — this might affect model training if we were to predict card behavior.

---

## 🚨 Step 5: Outlier Detection

Let’s use IQR (Interquartile Range) to find outliers in age, income, and fee.

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

- **Customer Age:** 11 outliers (older folks)
- **Income & Subscription Fee:** No extreme outliers

> ✅ That’s a relief — most of our numerical features look well-distributed!

---

## 🔗 Step 6: Correlation Analysis

```python
sns.heatmap(df[['Customer Age', 'Income', 'Subscription Fee']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Numerical Features')
plt.show()
```

### 🔍 Output: Correlation Heatmap
![plot4](https://github.com/user-attachments/assets/1cc12667-1be2-41d9-ab57-60817a36f814)

> 🔎 Not much correlation between age and income, but **subscription fee has a modest positive correlation with income**, which makes intuitive sense.

---

====
## 🔗 Step 7: Feature Importances Visualization

Here's how you can structure this step in the **same interactive, narrative style** as your original walkthrough, complete with simulated outputs and plots:


### 📌 **Goal**:  
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
plt.axhline(y=0, color='black', linestyle='--')  # Reference line
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

---

### 📊 **Output & Interpretation**

###### Numerical Correlation Coefficients
```
Correlation with Subscription Fee:
 Subscription Fee    1.000000
Income              0.933665
Customer Age        0.159394
Name: Subscription Fee, dtype: float64
```

- 💡 **Key Insight**:  
  - `Income` has a **very strong positive correlation (0.93)** with subscription fees.  
  - `Age` shows a **weak relationship (0.16)**, suggesting it’s a minor factor.  

---

###### Bar Plot: Correlation Strength**  
![plot7](https://github.com/user-attachments/assets/3af00a23-19e5-4017-bf7f-687ce80223c8)

- `Income` dominates the predictive power.  
- The dashed line at `y=0` helps quickly spot positive/negative relationships.  

---

###### Scatter Plots: Visual Trends**  
![plo8](https://github.com/user-attachments/assets/631ab1d7-d734-4201-b420-d6f4a930237e)

- **Left (Income)**:  
  - Clear upward trend: Higher income → Higher fees.  
  - Tight clustering suggests income is a **reliable predictor**.  
- **Right (Age)**:  
  - Weak trend with high variance: Age alone isn’t decisive.  

### 🎯 **Actionable Takeaways**  
1. **Focus on `Income`**: Strongest driver of subscription fees.  
2. **Question `Age`**: Weak correlation—consider engineering age groups (e.g., "Young/Middle/Senior") for better signal.  
3. **Next Step**: Use this insight to prioritize features for modeling (e.g., drop `Age` or transform it).  


---
## 🧠 Step 8: Feature Selection — *You're in Control!* 

Now that we’ve engineered some cool features, it’s time to zoom in on the **most important ones** — but here’s the twist: **you get to decide** how many!

### 👉 Why Feature Selection?
Feature selection helps:
- Remove irrelevant/noisy data
- Reduce overfitting
- Speed up training
- Improve model performance 🚀
![—Pngtree—blank button for web designing_6221303](https://github.com/user-attachments/assets/e7c17efa-8969-4812-93e5-1ac310d9a652)


> **Nice!** These are the features that statistically have the strongest relationship with the subscription fee tier. This helps reduce noise and improve model performance!

---


## 🔧 Step 9: Feature Engineering

This is where the magic starts! We’re creating dummies, weekend flags, and age groups.

```python
df = pd.get_dummies(df, columns=['Card Category', 'Weekday'], drop_first=True)

df['Is_Weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)

def age_group(age):
    if age < 25:
        return 'Young'
    elif age < 45:
        return 'Middle-Aged'
    else:
        return 'Senior'

df['Age_Group'] = df['Customer Age'].apply(age_group)
df = pd.get_dummies(df, columns=['Age_Group'], drop_first=True)
```

> 🎨 Now our data is rich with useful features, ready for modeling!


---
## 📦 Step 10: Preparing Data for ML

We're prepping for model training by selecting features and scaling them.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['Customer Name', 'Date', 'Subscription Fee'])
y = df['Subscription Fee']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> 📌 **Tip:** Always scale your numerical features before using models like Logistic Regression or SVM.

---

## 🎯 Step 11: Target Engineering — Fee Tiers

We're switching the problem from regression to classification by creating **fee tiers**.

```python
df['Fee_Tier'] = pd.qcut(df['Subscription Fee'], q=3, labels=[0, 1, 2]).astype(int)

X = df.drop(columns=['Customer Name', 'Date', 'Subscription Fee', 'Fee_Tier'])
y = df['Fee_Tier']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> 🏷️ This gives us a **3-class classification problem**: Low, Medium, and High fee customers — very useful for segment-based marketing!

---

## step 12: Model Training

> **Nice!** Now i will start generating model as teh next step!


🧠 **Coming up next**: we’ll train a model and evaluate it using classification metrics and a confusion matrix.

Ready to move into modeling? Just say the word, and I’ll walk you through Logistic Regression and beyond — step by step!
