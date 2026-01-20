# 2025aa05604 - Naresh Seth

# Machine Learning Assignment 2 — End-to-End Classification (Streamlit)

## a) Problem statement

Build an end-to-end classification system using **one public dataset** (binary or multi-class).  
Implement **6 classification models**, evaluate them using standard metrics, and deploy an interactive **Streamlit** app.

Required models:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Required metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- MCC Score

## b) Dataset description

This is a comprehensive Customer Engagement and Churn Analytics Dataset containing behavioral, demographic, and transactional data for 50,000 customers across a global e-commerce/subscription platform. The dataset captures 25 distinct features that provide a 360-degree view of customer interactions and engagement patterns.

### Dataset Characteristics

<https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset>

| Characteristic          | Details                                                   |
| :---------------------- | :-------------------------------------------------------- |
| **Records**             | 50,000 customers                                          |
| **Features**            | 25 columns                                                |
| **Data Types**          | Mixed (numerical, categorical, object)                    |
| **Geographic Coverage** | USA, UK, Germany, Canada, India, Japan, France, Australia |
| **Time Period**         | Customer journey from signup through current status       |

### Key Feature Categories

**1. Customer Demographics (5 features)**

- Age, Gender, Country, City, Membership_Years

**2. Platform Engagement (8 features)**

- Login_Frequency, Session_Duration_Avg, Pages_Per_Session
- Cart_Abandonment_Rate, Wishlist_Items, Email_Open_Rate
- Mobile_App_Usage, Social_Media_Engagement_Score

**3. Purchase Behavior (6 features)**

- Total_Purchases, Average_Order_Value, Days_Since_Last_Purchase
- Discount_Usage_Rate, Return_Rate, Payment_Method_Diversity

**4. Customer Service (3 features)**

- Customer_Service_Calls, Product_Reviews_Written, Lifetime_Value

**5. Financial & Status (3 features)**

- Credit_Balance, Churned (target variable), Signup_Quarter

### Target Variable

**Churned**: Binary indicator showing whether a customer has discontinued using the service (1) or remains active (0)

## c) Models used + Evaluation comparison table

### Model Performance Metrics

| Model               | Accuracy |  AUC   | Precision | Recall | F1 Score |  MCC   |
| :------------------ | :------: | :----: | :-------: | :----: | :------: | :----: |
| Logistic Regression |  0.7757  | 0.7905 |  0.6788   | 0.4249 |  0.5227  | 0.4041 |
| Decision Tree       |  0.8402  | 0.8105 |  0.7163   | 0.7401 |  0.7280  | 0.6151 |
| kNN                 |  0.8023  | 0.7944 |  0.7595   | 0.4623 |  0.5747  | 0.4796 |
| Naive Bayes         |  0.6466  | 0.6920 |  0.4216   | 0.5990 |  0.4949  | 0.2441 |
| Random Forest       |  0.9133  | 0.9247 |  0.9192   | 0.7675 |  0.8365  | 0.7840 |
| XGBoost             |  0.9239  | 0.9297 |  0.9291   | 0.7976 |  0.8583  | 0.8111 |

### Model Observations & Analysis

| Model               | Observations                                                                                                                                                                                       |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Simple linear baseline with moderate performance (77.6% accuracy). Good AUC (0.79) but lower recall (42.5%), suggesting it misses many churned customers. Best for interpretability.               |
| Decision Tree       | Strong performance (84% accuracy) with good balance between precision (71.6%) and recall (74%). Prone to overfitting but provides clear decision rules.                                            |
| kNN                 | Decent accuracy (80.2%) with high precision (75.95%) but low recall (46.2%). Instance-based approach works moderately; sensitive to feature scaling and k selection.                               |
| Naive Bayes         | Poorest performer (64.7% accuracy, 0.69 AUC). Low precision (42.2%) but moderate recall (59.9%). Assumption of feature independence poorly suited for this dataset.                                |
| Random Forest       | Excellent performance (91.3% accuracy, 0.924 AUC) with strong precision (91.9%) and recall (76.75%). Handles non-linearity well; low risk of overfitting due to ensemble nature.                   |
| XGBoost             | Best overall model (92.4% accuracy, 0.9297 AUC) with highest precision (92.9%) and excellent recall (79.76%). Gradient boosting effectively captures complex patterns. Recommended for production. |

## How to run locally

```bash
pip install -r requirements.txt
```

### Step 2: Train Models (One-time Setup)

Train all models on your full dataset and save them:

```bash
python model/train_offline.py --data data/ecommerce_customer_churn_dataset.csv --target Churned
```

**Arguments:**

- `--data` (required): Path to your training CSV dataset
- `--target` (required): Target column name in the dataset
- `--output` (optional): Output directory for saved models (default: `model/saved`)

**Output:**

- `model/saved/logreg_model.pkl` - Logistic Regression model
- `model/saved/dt_model.pkl` - Decision Tree model
- `model/saved/knn_model.pkl` - KNN model
- `model/saved/nb_model.pkl` - Naive Bayes model
- `model/saved/rf_model.pkl` - Random Forest model
- `model/saved/xgb_model.pkl` - XGBoost model
- `model/saved/preprocessor.pkl` - Feature preprocessor
- `model/saved/metadata.pkl` - Training metadata

### Step 3: Run the Streamlit App

```bash
streamlit run app.py
```

The app will:

1. Load pre-trained models from `model/saved/`
2. Accept a test CSV file (same features as training data)
3. Display evaluation metrics across all models
4. Show detailed analysis for selected model
5. Allow single-row predictions

---

## Project Structure

```
├── app.py                          # Streamlit web app (loads pre-trained models)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   └── ecommerce_customer_churn_dataset.csv  # Original training dataset
│
├── model/
│   ├── train_offline.py            # Training script (trains and saves models)
│   ├── train_models.py             # Model training logic
│   ├── evaluate.py                 # Evaluation metrics & visualizations
│   ├── utils.py                    # Data preprocessing utilities
│   │
│   └── saved/                      # Pre-trained models (created after training)
│       ├── logreg_model.pkl
│       ├── dt_model.pkl
│       ├── knn_model.pkl
│       ├── nb_model.pkl
│       ├── rf_model.pkl
│       ├── xgb_model.pkl
│       ├── preprocessor.pkl
│       └── metadata.pkl
```

---

## File Descriptions

| File                      | Purpose                                                        |
| ------------------------- | -------------------------------------------------------------- |
| `app.py`                  | Main Streamlit app - loads models and accepts test data        |
| `model/train_offline.py`  | Standalone script to train and save all models                 |
| `model/train_models.py`   | Core training logic for all 6 models                           |
| `model/evaluate.py`       | Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) |
| `model/utils.py`          | Data preprocessing (StandardScaler, OneHotEncoder)             |

---

## Workflow

### Training Phase (One-time)

```
Full Training Data
       ↓
train_offline.py (preprocesses & trains all 6 models)
       ↓
Saves: 6 models + preprocessor + metadata
       ↓
Pickled files in model/saved/
```

### Evaluation Phase (Repeated)

```
Test Data Upload (CSV)
       ↓
app.py (loads pre-trained models)
       ↓
Transform test data using saved preprocessor
       ↓
Evaluate & Display Results
       ↓
Show metrics, confusion matrix, classification report
```

---

## Models Used

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Tree-based model
3. **k-Nearest Neighbors (kNN)** - Instance-based learning
4. **Naive Bayes** - Probabilistic classifier
5. **Random Forest** - Ensemble (n_estimators=300)
6. **XGBoost** - Gradient boosting ensemble (n_estimators=400)

---

## Evaluation Metrics

- **Accuracy**: Proportion of correct predictions
- **AUC Score**: Area under ROC curve (multi-class: One-vs-Rest)
- **Precision**: True positives / All predicted positives
- **Recall**: True positives / All actual positives
- **F1 Score**: Harmonic mean of Precision and Recall
- **MCC**: Matthews Correlation Coefficient (balanced metric)

4. Deploy

## Project structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   │-- train_models.py
│   │-- evaluate.py
│   │-- utils.py
```

## ⚠️ Git LFS Notice

This repository uses **Git Large File Storage (LFS)** to handle large model files and datasets. If you clone this repository, ensure you have Git LFS installed:

```bash
git lfs install
```

Then clone the repository normally:

```bash
git clone https://github.com/sethnaresh/2025aa05604_ml_assignment2.git
cd 2025aa05604_ml_assignment2
```

Git LFS will automatically fetch large files. Without LFS installed, large files will appear as pointers instead of actual files.

**Files tracked with LFS:**

- `*.pkl` (Model files)
- `*.csv` (Datasets)
