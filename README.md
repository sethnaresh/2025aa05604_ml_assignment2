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

| Model               | Accuracy |  AUC  | Precision | Recall | F1 Score |  MCC  |
| :------------------ | :------: | :---: | :-------: | :----: | :------: | :---: |
| Logistic Regression |          |       |           |        |          |       |
| Decision Tree       |          |       |           |        |          |       |
| kNN                 |          |       |           |        |          |       |
| Naive Bayes         |          |       |           |        |          |       |
| Random Forest       |          |       |           |        |          |       |
| XGBoost             |          |       |           |        |          |       |

### Model Observations & Analysis

| Model               | Observations |
| :------------------ | :----------- |
| Logistic Regression |              |
| Decision Tree       |              |
| kNN                 |              |
| Naive Bayes         |              |
| Random Forest       |              |
| XGBoost             |              |

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this project to GitHub
2. Go to Streamlit Cloud → New App
3. Select repo / branch / `app.py`
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
