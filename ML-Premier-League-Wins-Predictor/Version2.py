import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # For saving models

# Start timer
start_time = time.time()

# Load data
league_data = pd.read_csv(r"C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\data\results.csv")
stats = pd.read_csv(r'C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\data\stats.csv')
standings = pd.read_csv(r'C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\data\EPL Standings 2000-2022.csv')
with_goalscorers = pd.read_csv(r'C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\data\with_goalscorers.csv')

# Assertions for initial data checks
assert league_data is not None, "League data not loaded."
assert stats is not None, "Stats data not loaded."
assert standings is not None, "Standings data not loaded."
assert with_goalscorers is not None, "Goalscorer data not loaded."

# Export data to Excel for review
league_data.to_excel("league_data.xlsx", index=False)

# Ensure expected structure of league_data
expected_columns = ['home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'season']
assert list(league_data.columns) == expected_columns, "Unexpected columns in league data."

# Filter standings data between specific seasons
new_standings = standings[standings['Season'].between('2006-07', '2017-18')]
new_standings = new_standings.drop('Qualification or relegation', axis=1).reset_index(drop=True)

# Extract Top Scorer details in goalscorer data
with_goalscorers[['Top Scorer Goals', 'Top Scorer Team']] = with_goalscorers['Top Scorer'].str.extract(r'(\d+)\((.*?)\)$')
with_goalscorers['Top Scorer Goals'] = with_goalscorers['Top Scorer Goals'].astype(int)
with_goalscorers_sorted = with_goalscorers[with_goalscorers.Season.between('2006-2007', '2017-2018')].reset_index(drop=True)

# Data Analysis
champions = new_standings[new_standings.Pos == 1].reset_index(drop=True)
second_placed = new_standings[new_standings.Pos == 2].reset_index(drop=True)
winning_margin = champions.Pts - second_placed.Pts
champions = champions.assign(Winning_margin=winning_margin.values)

# Plot Winning Margin Over Seasons
plt.plot(champions.Season, champions.Winning_margin)
plt.xlabel('Season')
plt.ylabel('Winning margin')
plt.xticks(rotation=45)
plt.title('Winning Margin Over Seasons (2006-07 to 2017-18)')
plt.show()

# Titles won distribution pie chart
times_champion = champions.Team.value_counts()
teams = times_champion.index.to_list()
plt.pie(times_champion, labels=teams, autopct='%1.1f%%', colors=['red', 'royalblue', 'skyblue', 'white'])
plt.title('Titles won from 2006-07 to 2017-18')
plt.show()

# Prepare data for modeling
y = stats['wins']
X = stats.drop(['wins', 'season', 'team'], axis=1)

# Handle missing values
try:
    imputer = joblib.load(r'C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\imputer.pkl')
except FileNotFoundError:
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    # Save the imputer model
    joblib.dump(imputer, r'C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\imputer.pkl')
else:
    X_imputed = imputer.transform(X)

# Scaling data
try:
    scaler = joblib.load(r'C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\scaler.pkl')
except FileNotFoundError:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    # Save the scaler model
    joblib.dump(scaler, r'C:\Users\Windows\Desktop\ML_EPL_Win_predict\ML-Premier-League-Wins-Predictor\scaler.pkl')
else:
    X_scaled = scaler.transform(X_imputed)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Linear Regression Evaluation
mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
print("Linear Regression Metrics:")
print(f"MAE: {mae_lin:.2f}, MSE: {mse_lin:.2f}, RMSE: {rmse_lin:.2f}")

# Save the Linear Regression model
joblib.dump(lin_reg, 'linear_regression_model.joblib')

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Logistic Regression Evaluation with zero_division handling
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log, average='weighted', zero_division=1)
recall_log = recall_score(y_test, y_pred_log, average='weighted', zero_division=1)
f1_log = f1_score(y_test, y_pred_log, average='weighted', zero_division=1)
print("\nLogistic Regression Metrics:")
print(f"Accuracy: {accuracy_log:.2f}, Precision: {precision_log:.2f}, Recall: {recall_log:.2f}, F1 Score: {f1_log:.2f}")

# Save the Logistic Regression model
joblib.dump(log_reg, 'logistic_regression_model.joblib')

# Random Forest Regressor Model
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# Random Forest Evaluation
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
print("\nRandom Forest Regressor Metrics:")
print(f"MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, RMSE: {rmse_rf:.2f}")

# Save the Random Forest model
joblib.dump(rf_reg, 'random_forest_regressor_model.joblib')

# Feature Importance for Linear Regression
feature_importance = pd.Series(lin_reg.coef_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance (Linear Regression):")
print(feature_importance)

# Visualization for Actual vs. Predicted Wins (Linear Regression)
actual_pred_lin = pd.DataFrame({'Actual wins': y_test, 'Predicted wins': y_pred_lin}).reset_index(drop=True)
plt.plot(actual_pred_lin.index, actual_pred_lin['Actual wins'], label='Actual Wins')
plt.plot(actual_pred_lin.index, actual_pred_lin['Predicted wins'], label='Predicted Wins')
plt.xlabel('Index')
plt.ylabel('Wins')
plt.title('Actual vs. Predicted Wins (Linear Regression)')
plt.legend()
plt.show()

# Elapsed time
end_time = time.time()
print(f"\n[Elapsed time: {end_time - start_time:.2f} seconds]")
