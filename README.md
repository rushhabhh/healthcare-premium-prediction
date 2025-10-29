# Healthcare Premium Prediction

## Overview  
This project predicts the **health insurance premium cost** for individuals based on their demographic and lifestyle attributes using **machine learning regression models**.  

Accurate premium estimation enables insurance companies to **assess customer risk profiles** and **set fair premium prices**, while helping individuals understand how factors like age, BMI, and smoking habits affect their insurance cost.

## Objectives
- Develop a machine learning model to predict **medical insurance premiums**.  
- Identify key factors influencing insurance cost (e.g., smoking, BMI, age).  
- Compare regression algorithms to find the best performing model.  

## Dataset
- **Features:**
  - `age`: Age of the individual  
  - `sex`: Gender (male/female)  
  - `bmi`: Body Mass Index  
  - `children`: Number of dependents  
  - `smoker`: Smoking status (yes/no)  
  - `region`: Residential region  
  - `charges`: Medical insurance premium (target variable)

## Methodology

### 1. Data Preprocessing
- Checked for missing values and outliers  
- Encoded categorical variables (`sex`, `smoker`, `region`) using one-hot encoding  
- Scaled continuous features (`age`, `bmi`, `children`)  
- Split dataset into **training and testing sets (80-20)**  

### 2. Exploratory Data Analysis (EDA)
- Analyzed correlation between independent variables and insurance charges  
- Visualized the impact of BMI, age, and smoker status using boxplots and heatmaps  
- Found that **smoking** and **BMI** were the strongest predictors of higher premiums  

### 3. Model Building
Implemented and compared multiple regression models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- XGBoost Regressor  

### 4. Model Evaluation
Evaluated each model using:
- **Mean Absolute Error (MAE)**  
- **Root Mean Squared Error (RMSE)**  
- **R² Score (Coefficient of Determination)**  

---

## 📊 Results
| Model | MAE | RMSE | R² Score |
|--------|-----|------|----------|
| Linear Regression | 4100 | 5800 | 0.74 |
| Decision Tree | 1900 | 2900 | 0.86 |
| Random Forest | 1300 | 2500 | 0.90 |
| XGBoost | **1200** | **2300** | **0.92** |

> **XGBoost Regressor** achieved the best performance with the lowest error and highest R² score.

## Key Insights
- **Smoking** status is the most critical factor affecting premiums — smokers pay up to 4× higher charges.  
- **BMI** and **age** also significantly influence medical costs.  
- Ensemble models like Random Forest and XGBoost provide superior performance due to their ability to capture nonlinear relationships.  

## Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost  
- **Environment:** Jupyter Notebook / Google Colab  

## How to Run
```bash
# Clone this repository
git clone https://github.com/<your-username>/healthcare-premium-prediction.git

# Navigate to the project folder
cd healthcare-premium-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook Healthcare_Premium_Prediction.ipynb

Also Playaround with the deployed app: https://healthcare-premium-pred-ml.streamlit.app/
