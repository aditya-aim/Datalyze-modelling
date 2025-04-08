# 📚 Datalyze RAG Flask API

## 📑 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [Clone Repository](#1-clone-the-repository)
  - [Set Up Qdrant](#2-set-up-qdrant-with-docker)
  - [Set Up Environment](#3-set-up-the-environment)
  - [Ingest Data](#4-ingest-data)
  - [Run Flask App](#5-run-flask-app)
- [API Endpoints](#-api-endpoints)
  - [RAG Endpoints](#rag-endpoints)
    - [Document Retrieval](#1-document-retrieval)
    - [Answer Generation](#2-answer-generation)
    - [Comparative Analysis](#3-comparative-analysis)
  - [Modeling Endpoints](#modeling-endpoints)
    - [Get Schema](#1-get-schema)
    - [Train Models](#2-train-models)
    - [Make Predictions (Base Model)](#3-make-predictions-base-model)
    - [Fine-tune Model](#4-fine-tune-model)
    - [Make Predictions (Tuned Model)](#5-make-predictions-tuned-model)
- [Expected Responses](#expected-responses)
- [Testing with Postman](#-testing-with-postman)
- [Contributing](#-contributing)

## 🌟 Features

- **Data Ingestion**: Ingest data from PDF documents and store it in Qdrant Cloud DB.
- **Information Retrieval**: Retrieve relevant information using vector search in Qdrant based on the user's query.
- **Answer Generation**: Generate concise answers by augmenting the retrieved context with a language model.
- **Machine Learning Modeling**: Train, fine-tune, and make predictions using various ML models.
- **Modular Design**: Each component of the RAG model is encapsulated in its own Flask blueprint, ensuring a clean and maintainable codebase.
- **Easy Deployment**: Ready-to-deploy with environment configurations for different setups.

## 📂 Project Structure

```bash
datalyze-rag-flask-api/
├── api/
│   ├── __init__.py             
│   ├── common.py               
│   ├── comparative_analysis.py  # Comparative analysis using prompt templates
│   ├── generation.py           # Answer generation using LLM
│   ├── ingestion.py            # Data ingestion into Qdrant
│   ├── retrieval.py            # Vector search and retrieval
│   ├── modeling.py             # ML modeling endpoints
│   ├── dataset_utils.py        # Common dataset utilities
│   ├── dataset1.py             # Dataset1 specific implementations
│   ├── datasets/               # Dataset files
│   │   └── d1.realistic_subscription_data.csv
│   └── models/                 # Trained model files
│       └── d1.realistic_subscription_models/
├── data/
│   └── data pdfs               # PDF documents for RAG
├── config.py                   # Configuration settings
├── app.py                      # Main Flask application
├── ingest_data.py              # Data ingestion script
├── requirements.txt            # Python dependencies
├── postman-modeling.json       # Postman collection for modeling endpoints
├── README.md                   # Project documentation
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/analyticsindiamag/datalyze-rag-flask-api
cd datalyze-rag-flask-api
```

### 2. Set Up Qdrant with Docker

* Install Docker if you haven't already (visit [Docker's official website](https://www.docker.com/get-started))
* Pull and run the Qdrant container:

```bash
# Pull the latest Qdrant image
docker pull qdrant/qdrant

# Run Qdrant container
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

* Verify Qdrant is running:
```bash
curl http://localhost:6333/healthz
```

### 3. Set Up the Environment

* Create and activate a virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate  # On Linux use `venv\bin\activate`
```

* Install the dependencies:

```bash
pip install -r requirements.txt
```

* Create a `.env` file in the project root with your configurations:

```bash
# .env
QDRANT_URL=http://localhost:6333  # Local Qdrant instance URL
QDRANT_API_KEY=                   # Leave empty for local instance
GROQ_API_KEY=<Your-Groq-API-Key>  # Required for answer generation
DEBUG=True
```

### 4. Ingest Data

Run the `ingest_data.py` script to ingest data into Qdrant:

```bash
python ingest_data.py
```

### 5. Run Flask App

```bash
python app.py
```

## 📱 API Endpoints

### RAG Endpoints

#### 1. Document Retrieval
* **Endpoint**: `POST /api/retrieval/retrieve`
* **Headers**: 
  - Content-Type: application/json
* **Body** (raw JSON):
```json
{
    "query": "What is age estimation?"
}
```

#### 2. Answer Generation
* **Endpoint**: `POST /api/generation/generate`
* **Headers**: 
  - Content-Type: application/json
* **Body** (raw JSON):
```json
{
    "query": "Explain the process of age estimation"
}
```

#### 3. Comparative Analysis
* **Endpoint**: `POST /api/comparative/compare`
* **Headers**: 
  - Content-Type: application/json
* **Body** (raw JSON):
```json
{
    "comparison_query": "Compare credit card spending patterns between different age groups"
}
```

### Modeling Endpoints

The Modeling API provides a comprehensive set of endpoints for machine learning operations, including model training, prediction, and fine-tuning. These endpoints are designed to work with different datasets, with dataset1 (realistic subscription data) currently supported.

#### 1. Get Schema
* **Endpoint**: `GET /api/modeling/schema/<dataset_id>`
* **Description**: Get input schema for a specific dataset, including field descriptions, data types, and valid values
* **Parameters**:
  - `dataset_id`: Identifier for the dataset (e.g., "d1" for realistic subscription data)
* **Example**: `GET /api/modeling/schema/d1`
* **Response**:
```json
{
    "status": "success",
    "base_predict": [
        {
            "key": "Card Category",
            "label": "Card Category",
            "type": "string",
            "options": [
                {"key": "Blue", "label": "Blue"},
                {"key": "Gold", "label": "Gold"},
                {"key": "Platinum", "label": "Platinum"},
                {"key": "Silver", "label": "Silver"}
            ]
        },
        {
            "key": "Subscription Fee",
            "label": "Subscription Fee",
            "type": "number",
            "min": 0,
            "max": 1000,
            "step": 1
        }
        // Additional fields...
    ],
    "tune_form": [
        // Tuning form fields...
    ],
    "tune_parameters": {
        "RandomForest": {
            "n_estimators": {
                "type": "number",
                "label": "Number of Estimators",
                "min": 50,
                "max": 500,
                "step": 50,
                "default": 100
            },
            "max_depth": {
                "type": "number",
                "label": "Maximum Depth",
                "min": 1,
                "max": 20,
                "step": 1,
                "default": 10
            }
            // Additional parameters...
        }
        // Additional models...
    },
    "model_names": {
        "RandomForest_model": "Random Forest",
        "GradientBoosting_model": "Gradient Boosting",
        "LogisticRegression_model": "Logistic Regression",
        "SVM_model": "Support Vector Machine"
    }
}
```

#### 2. Train Models
* **Endpoint**: `POST /api/modeling/train/<dataset_id>`
* **Description**: Train and evaluate models for a specific dataset. This endpoint will train multiple models (Random Forest, Gradient Boosting, Logistic Regression, SVM) and return their performance metrics.
* **Parameters**:
  - `dataset_id`: Identifier for the dataset (e.g., "d1" for realistic subscription data)
* **Example**: `POST /api/modeling/train/d1`
* **Response**:
```json
{
    "status": "success",
    "models": {
        "RandomForest_model": {
            "accuracy": 0.795,
            "confusion_matrix": [[120, 15, 10], [8, 130, 12], [5, 8, 140]],
            "classification_report": {
                "Decrease": {"precision": 0.902, "recall": 0.828, "f1-score": 0.864},
                "Neutral": {"precision": 0.850, "recall": 0.867, "f1-score": 0.858},
                "Increase": {"precision": 0.865, "recall": 0.915, "f1-score": 0.889}
            }
        },
        "GradientBoosting_model": {
            "accuracy": 0.780,
            "confusion_matrix": [[115, 20, 10], [10, 125, 15], [8, 10, 135]],
            "classification_report": {
                "Decrease": {"precision": 0.865, "recall": 0.793, "f1-score": 0.828},
                "Neutral": {"precision": 0.806, "recall": 0.833, "f1-score": 0.819},
                "Increase": {"precision": 0.844, "recall": 0.882, "f1-score": 0.863}
            }
        }
        // Additional models...
    }
}
```

#### 3. Make Predictions (Base Model)
* **Endpoint**: `POST /api/modeling/predict/<dataset_id>/<model_name>`
* **Description**: Make predictions using a pre-trained base model for a specific dataset
* **Parameters**:
  - `dataset_id`: Identifier for the dataset (e.g., "d1" for realistic subscription data)
  - `model_name`: Name of the model to use (e.g., "RandomForest_model", "GradientBoosting_model")
* **Headers**: 
  - Content-Type: application/json
* **Body** (raw JSON):
```json
{
    "Card Category": "Gold",
    "Subscription Fee": 150,
    "Customer Age": 35,
    "Income": 75000,
    "Month": "June",
    "Customer Tenure": 24,
    "Marketing Channel": "Email"
}
```
* **Example**: `POST /api/modeling/predict/d1/RandomForest_model`
* **Response**:
```json
{
    "status": "success",
    "prediction": 1,
    "prediction_label": "Increase",
    "probabilities": {
        "Decrease": 0.1,
        "Neutral": 0.2,
        "Increase": 0.7
    },
    "model_used": "RandomForest_model",
    "dataset": "d1"
}
```

#### 4. Fine-tune Model
* **Endpoint**: `POST /api/modeling/fine-tune/<dataset_id>/<model_name>`
* **Description**: Fine-tune a specific model with custom hyperparameters to improve its performance
* **Parameters**:
  - `dataset_id`: Identifier for the dataset (e.g., "d1" for realistic subscription data)
  - `model_name`: Name of the model to fine-tune (e.g., "RandomForest_model", "GradientBoosting_model")
* **Headers**: 
  - Content-Type: application/json
* **Body** (raw JSON):
```json
{
    "n_estimators": 150,
    "max_depth": 10,
    "min_samples_split": 5
}
```
* **Example**: `POST /api/modeling/fine-tune/d1/RandomForest_model`
* **Response**:
```json
{
    "status": "success",
    "accuracy": 0.815,
    "confusion_matrix": [[125, 10, 10], [5, 135, 10], [5, 5, 145]],
    "classification_report": {
        "Decrease": {"precision": 0.926, "recall": 0.862, "f1-score": 0.893},
        "Neutral": {"precision": 0.900, "recall": 0.900, "f1-score": 0.900},
        "Increase": {"precision": 0.879, "recall": 0.935, "f1-score": 0.906}
    },
    "model_path": "api/models/d1.realistic_subscription_models/RandomForest_tuned.pkl",
    "best_params": {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5
    }
}
```

#### 5. Make Predictions (Tuned Model)
* **Endpoint**: `POST /api/modeling/predict-tuned/<dataset_id>/<model_name>`
* **Description**: Make predictions using a fine-tuned model for a specific dataset
* **Parameters**:
  - `dataset_id`: Identifier for the dataset (e.g., "d1" for realistic subscription data)
  - `model_name`: Name of the tuned model to use (e.g., "RandomForest_tuned", "GradientBoosting_tuned")
* **Headers**: 
  - Content-Type: application/json
* **Body** (raw JSON):
```json
{
    "Card Category": "Gold",
    "Subscription Fee": 150,
    "Customer Age": 35,
    "Income": 75000,
    "Month": "June",
    "Customer Tenure": 24,
    "Marketing Channel": "Email"
}
```
* **Example**: `POST /api/modeling/predict-tuned/d1/RandomForest_tuned`
* **Response**:
```json
{
    "status": "success",
    "prediction": 1,
    "prediction_label": "Increase",
    "probabilities": {
        "Decrease": 0.08,
        "Neutral": 0.17,
        "Increase": 0.75
    },
    "model_used": "RandomForest_tuned",
    "dataset": "d1",
    "tuning_params": {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5
    }
}
```

### Expected Responses

#### Success Response
```json
{
    "status": "success",
    "prediction": 1,
    "prediction_label": "Increase",
    "probabilities": {
        "Decrease": 0.1,
        "Neutral": 0.2,
        "Increase": 0.7
    }
}
```

#### Error Response
```json
{
    "status": "error",
    "message": "Error description"
}
```

## 📱 Testing with Postman

### Setting up Postman

1. Download and install [Postman](https://www.postman.com/downloads/)
2. Import the `postman-modeling.json` collection
3. Set the base URL to: `http://localhost:5030`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have suggestions or improvements.
