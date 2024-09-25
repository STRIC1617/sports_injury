import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Create a directory for saving models and figures
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load data
df = pd.read_csv('dataset/player_injury_data.csv')

# Check and handle NaN values
print("Checking for NaN values in the dataset...")
if df.isnull().sum().sum() > 0:
    print("Dropping NaN values...")
    df.dropna(inplace=True)

# Calculate average recovery time
df['avg_recovery_time'] = df.groupby('club')['days'].transform('mean')

# Preprocess features
df['type'] = LabelEncoder().fit_transform(df['type'])
df['bmi'] = df['weight'] / (df['height'] ** 2)

# Prepare feature set and target variable
X = df[['weight', 'height', 'club_value', 'age', 'type', 'bmi', 'avg_recovery_time']]
y = df['days']

# Ensure all data types are numeric
X = X.apply(pd.to_numeric, errors='coerce').dropna() # Data
y = pd.to_numeric(y, errors='coerce')[X.index] #Labels

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models for ensemble
base_models = [
    ('linear', LinearRegression()),
    ('decision_tree', DecisionTreeRegressor()),
    ('random_forest', RandomForestRegressor()),
    ('svr', SVR()),
    ('knn', KNeighborsRegressor())
]

# Create ensemble models
models = {
    'Voting Regressor': VotingRegressor(estimators=base_models),
    'Stacking Regressor': StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
}

# Store metrics for plotting
metrics = []

# Evaluate and save models
for name, model in models.items():
    model.fit(X_train, y_train) # Data train
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics.append({'Model': name, 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R^2': r2})
    joblib.dump(model, f'models/{name.replace(" ", "_").lower()}_ensemble_model.pkl')

# Convert metrics to DataFrame for visualization
metrics_df = pd.DataFrame(metrics)

# Set the style
sns.set(style='whitegrid')

# Plotting the metrics
for metric in ['MSE', 'MAE', 'RMSE', 'R^2']:
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model', y=metric, data=metrics_df, palette='viridis')
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'figures/{metric.lower()}_comparison.png')  # Save each figure
    plt.close()

# Determine the best model based on MSE
best_model = metrics_df.loc[metrics_df['MSE'].idxmin()]
print(f"\nBest Model: {best_model['Model']} with MSE = {best_model['MSE']:.2f}")
