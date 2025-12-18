# Energy Consumption Prediction Model 🔌

The goal of this project is to predict **energy consumption** based on building characteristics, occupancy, appliance usage, and environmental factors.

---

## 📌 Problem Statement
Energy consumption varies significantly depending on building type, size, occupancy, appliances used, and external conditions such as temperature and day of the week.  
This project aims to build a regression-based machine learning model to accurately predict energy consumption using these factors.

---

## 📂 Dataset Features
- Building Type  
- Square Footage  
- Number of Occupants  
- Appliances Used  
- Average Temperature  
- Day of Week  
- Energy Consumption (Target Variable)

---

## 🛠️ Tech Stack & Tools
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- Joblib  
- Git & GitHub  

---

## 🔍 Exploratory Data Analysis (EDA)
Performed detailed EDA to understand:
- Data distribution and skewness
- Relationship between features and target variable
- Categorical vs numerical feature impact
- Outliers and variability in energy consumption

### Visualizations Used:
- Histogram & KDE plots
- Scatter plots
- Box plots & Violin plots
- Bar plots & Count plots
- Heatmap (correlation analysis)
- Pair plots (numerical feature relationships)

---

## ⚙️ Data Preprocessing
- Split data into training and testing sets
- Label encoding for categorical features
- Feature scaling using MinMaxScaler
- Maintained raw and processed data separation for reproducibility

---

## 🤖 Models Implemented
Since the data showed **slight non-normal distribution**, multiple regression models were trained and compared:

- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

---

## 📈 Model Evaluation
Models were evaluated using:
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The comparison helped identify the best-performing model for energy consumption prediction.

---

## 👤 Author

**Shivam Singh**
Aspiring Data Scientist | Machine Learning Enthusiast

🔗 GitHub: [https://github.com/shivamsingh-itds]
🔗 LinkedIn: [www.linkedin.com/in/shivamsinghds]

---

⭐ If you find this project helpful, feel free to star the repository!
