
---

# 💳 Credit Card Spending Analysis

## 👋 Introduction
Hey there! We're analyzing credit card transactions to predict high-spending customers. Let's dive in!

```python
import pandas as pd
data = pd.read_csv('d7.Decline_Credit_Card_Spending.csv')
data.head()
```

### Sample Data:
| Transaction Date | Age Group | Amount | Category | Reward Points |
|------------------|-----------|--------|----------|---------------|
| 2023-04-13       | 18-25     | 18.45  | Groceries | 13            |

🔹 **Features**:
- Customer Age Group (Categorical)  
- Transaction Category  
- Reward Points (Numeric)  
- Transaction Frequency  

🔹 **Target**: `High Spender` (1 = Above median, 0 = Below)

---

## 🔍 Step 1: Exploring Our Data

```python
# Create target variable
median = data['Transaction Amount'].median()
data['High Spender'] = (data['Transaction Amount'] > median).astype(int)

# Check distributions
print(data['High Spender'].value_counts())
sns.histplot(data['Transaction Amount'], kde=True)
```

![Histogram](https://github.com/user-attachments/assets/4f326bb7-bc0a-4887-9d79-8d6c34a6d5a3)

### Key Findings:
- Perfectly balanced target (50/50)  
- Right-skewed transaction amounts  
- No missing values 🎯  

---

## 📊 Step 2: Feature Relationships

```python
# Encode categoricals
data_encoded = pd.get_dummies(data, columns=['Age Group', 'Category'])

# Correlation analysis
corr = data_encoded.corr()['High Spender'].sort_values()
sns.barplot(x=corr.values, y=corr.index, palette="viridis")
```

![Correlation](https://github.com/user-attachments/assets/36828523-f000-46c8-830f-7272a9feea31)

### Insights:
- **Strong correlation**: Transaction Amount (0.82)  
- **Moderate**: Transaction Frequency (0.43)  
- **Age**: 41+ customers spend more  

🚨 **Warning**: Transaction Amount may cause target leakage!

#### Target Variable:
- `High Spender`

#### Relevant Features:
- Transaction Amount  
- Reward Points  
- Age Group_41+

#### All Features:
- Transaction Amount  
- Reward Points  
- Age Group_<18  
- Age Group_41+  
- Category_Travel  

---

## 🤖 Step 3: Choosing Our Models

### Top Contenders:
| Model               | Strength                       |
|--------------------|--------------------------------|
| Logistic Regression | Interpretable baseline         |
| Random Forest       | Handles non-linear patterns    |
| XGBoost             | State-of-the-art accuracy      |
| SVM                 | Works well with many features  |

```python
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}
```

### Evaluation Metrics:
- Accuracy  
- Precision/Recall  
- ROC-AUC  

#### Available Models:
- Logistic Regression  
- Random Forest  
- XGBoost  
- SVM  
- KNN  

#### Selected Models:
- Logistic Regression  
- Random Forest  
- XGBoost  

---

## 🚀 Step 4: Ready for Modeling!

We'll now:
1. Split data (80% train, 20% test)  
2. Scale numerical features  
3. Train selected models  
4. Evaluate performance  

![Modeling Process](https://github.com/user-attachments/assets/e7c17efa-8969-4812-93e5-1ac310d9a652)

🔁 **Iteration Options**:
- Tune hyperparameters  
- Try different feature combinations  
- Compare model results  

Let’s build some powerful predictors!

--- 

