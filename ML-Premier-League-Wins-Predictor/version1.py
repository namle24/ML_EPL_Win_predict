import numpy as np
import pandas as pd
import unittest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.league_data = pd.read_csv("data/results.csv")
        cls.stats = pd.read_csv('data/stats.csv')
        cls.standings = pd.read_csv('data/EPL Standings 2000-2022.csv')
        cls.with_goalscorers = pd.read_csv('data/with_goalscorers.csv')

    def test_league_data_not_none(self):
        self.assertIsNotNone(self.league_data)

    def test_stats_not_none(self):
        self.assertIsNotNone(self.stats)

    def test_standings_not_none(self):
        self.assertIsNotNone(self.standings)

    def test_with_goalscorers_not_none(self):
        self.assertIsNotNone(self.with_goalscorers)

    def test_league_data_columns(self):
        expected_columns = ['home_team', 'away_team', 'home_goals', 'away_goals', 'result', 'season']
        self.assertEqual(list(self.league_data.columns), expected_columns)

    def test_league_data_shape(self):
        expected_shape = (4560, 6)
        self.assertEqual(self.league_data.shape, expected_shape)

    def test_new_standings_shape(self):
        new_standings = self.standings[self.standings['Season'].between('2006-07', '2017-18')]
        new_standings.reset_index(drop=True, inplace=True)
        new_standings = new_standings.drop('Qualification or relegation', axis=1)
        self.assertEqual(new_standings.shape, (240, 11))

    def test_with_goalscorers_sorted_shape(self):
        with_goalscorers = self.with_goalscorers.copy()
        with_goalscorers[['Top Scorer Goals', 'Top Scorer Team']] = with_goalscorers['Top Scorer'].str.extract(r'(\d+)\((.*?)\)$')
        with_goalscorers.drop('Top Scorer', axis=1, inplace=True)
        with_goalscorers['Top Scorer Goals'] = with_goalscorers['Top Scorer Goals'].astype(int)
        with_goalscorers.drop('# Squads', axis=1, inplace=True)
        with_goalscorers_sorted = with_goalscorers[with_goalscorers.Season.between('2006-2007', '2017-2018')]
        self.assertEqual(with_goalscorers_sorted.shape, (12, 5))

    def test_champions_shape(self):
        new_standings = self.standings[self.standings['Season'].between('2006-07', '2017-18')]
        new_standings.reset_index(drop=True, inplace=True)
        new_standings = new_standings.drop('Qualification or relegation', axis=1)
        champions = new_standings[new_standings.Pos == 1]
        champions.reset_index(drop=True, inplace=True)
        self.assertEqual(champions.shape, (12, 11))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# Data Analysis and Plotting
league_data = pd.read_csv("data/results.csv")
stats = pd.read_csv('data/stats.csv')
standings = pd.read_csv('data/EPL Standings 2000-2022.csv')
with_goalscorers = pd.read_csv('data/with_goalscorers.csv')

new_standings = standings[standings['Season'].between('2006-07', '2017-18')]
new_standings.reset_index(drop=True, inplace=True)
new_standings = new_standings.drop('Qualification or relegation', axis=1)

with_goalscorers[['Top Scorer Goals', 'Top Scorer Team']] = with_goalscorers['Top Scorer'].str.extract(r'(\d+)\((.*?)\)$')
with_goalscorers.drop('Top Scorer', axis=1, inplace=True)
with_goalscorers['Top Scorer Goals'] = with_goalscorers['Top Scorer Goals'].astype(int)
with_goalscorers.drop('# Squads', axis=1, inplace=True)
with_goalscorers_sorted = with_goalscorers[with_goalscorers.Season.between('2006-2007', '2017-2018')]

champions = new_standings[new_standings.Pos == 1]
champions.reset_index(drop=True, inplace=True)

most_goals_season = stats.loc[stats.goals.idxmax()][['team', 'goals', 'season']]
print(most_goals_season)

non_champions = new_standings[new_standings['Pos'] != 1]
non_champions.reset_index(drop=True, inplace=True)
most_goals_non_champion = non_champions.nlargest(1, 'GF')
print(most_goals_non_champion[['Team', 'GF', 'Season', 'Pts']])

champions_scored_less_1 = champions[champions['GF'] < non_champions['GF'].max()]
champions_scored_more = champions[champions.GF > non_champions.GF.max()]
print(champions_scored_more)

goalscorer_champions = with_goalscorers_sorted[with_goalscorers_sorted.apply(lambda row: str(row['Champion']) in str(row['Top Scorer Team']), axis=1)]

# Check if the slice is empty to avoid division by zero
sliced_data = with_goalscorers_sorted[14:26]
if len(sliced_data) > 0:
    percentage_tg_ch = len(goalscorer_champions) / len(sliced_data) * 100
    percentage_formatted = f"{percentage_tg_ch:.1f}%"
else:
    percentage_formatted = "N/A"

print(percentage_formatted)

second_placed = new_standings[new_standings.Pos == 2]
second_placed.reset_index(drop=True, inplace=True)
winning_margin = champions.Pts - second_placed.Pts
champions = champions.assign(Winning_margin=winning_margin.values)
print(champions)

# Plots
plt.plot(champions.Season, champions.Winning_margin)
plt.xlabel('Season')
plt.ylabel('Winning margin')
plt.xticks(rotation=45)
plt.show()

times_champion = champions.Team.value_counts()
teams = times_champion.index.to_list()
plt.pie(times_champion, labels=teams, autopct='%1.1f%%', colors=['red', 'royalblue', 'skyblue', 'white'])
plt.title('Titles won from 2006-07 to 2017-18')
plt.show()

# Modeling
y = stats['wins']
X = stats.drop(['wins', 'season', 'team'], axis=1)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Fit the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

actual_predicted = pd.DataFrame({'Actual wins': y_test.squeeze(), 'Predicted wins': y_pred.squeeze()}).reset_index(drop=True)
print(actual_predicted.head())

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}\n\n')

# Fit Logistic Regression model
logistic_regressor = LogisticRegression()
logistic_regressor.fit(X_train, y_train)

# Predict on the test set using Logistic Regression
y_pred_logistic = logistic_regressor.predict(X_test)

# Calculate performance metrics for Logistic Regression
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic, average='weighted')
recall_logistic = recall_score(y_test, y_pred_logistic, average='weighted')
f1_logistic = f1_score(y_test, y_pred_logistic, average='weighted')

# Print evaluation metrics for Logistic Regression
print("Logistic Regression Evaluation Metrics:")
print(f"Accuracy: {accuracy_logistic:.2f}")
print(f"Precision: {precision_logistic:.2f}")
print(f"Recall: {recall_logistic:.2f}")
print(f"F1 Score: {f1_logistic:.2f}")
print()

# Print evaluation metrics for Linear Regression
print("Linear Regression Evaluation Metrics:")
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}\n\n')

# Create a line plot for actual wins
plt.plot(actual_predicted.index, actual_predicted['Actual wins'], label='Actual Wins')

# Create a line plot for predicted wins
plt.plot(actual_predicted.index, actual_predicted['Predicted wins'], label='Predicted Wins')

plt.xlabel('Index')
plt.ylabel('Wins')
plt.title('Actual vs. Predicted Wins')
plt.legend()
plt.show()

# Plot the scatter plot
sns.scatterplot(x='Predicted wins', y='Actual wins', data=actual_predicted)

# Add the regression line
sns.regplot(x='Predicted wins', y='Actual wins', data=actual_predicted, scatter=False, color='red')

# Set the axis labels and title
plt.xlabel('Predicted wins')
plt.ylabel('Actual wins')
plt.title('Actual vs Predicted Wins')
plt.show()

# Feature importance
feature_importance = pd.Series(regressor.coef_, index=X.columns).sort_values(ascending=False)
print(feature_importance)
