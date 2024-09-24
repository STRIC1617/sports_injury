# Player Injury Analysis and Prediction

This project analyzes player injury data to understand injury trends and predict recovery times using various regression models. It combines data processing, visualization, and machine learning techniques to provide insights into player injuries in sports.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Models](#models)
- [Visualizations](#visualizations)
- [License](#license)

## Project Overview

The main goals of this project are to:
- Analyze the relationships between player characteristics and injury recovery times.
- Visualize injury trends.
- Build and evaluate regression models to predict recovery times based on player attributes and injury types.

## Installation

To run this project, you need to have Python 3.x installed. You can set up a virtual environment and install the required packages as follows:

```bash```
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn joblib Flask


Make sure to also download the dataset player_injury_data.csv and place it in the dataset folder.

## Usage
# Data Processing and Model Training:

Run the model_training.py script to preprocess the data, train regression models, and save the trained models and metrics visualizations.

python model_training.py


# Web Application:

The Flask web app allows users to input player data and predict injury recovery times.
Run the app with:

python app.py


## Data Sources
The dataset used for this analysis includes player injury data, player characteristics, and club statistics. It can be found in the dataset directory.
# Models
This project employs several regression models for predicting recovery times:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor (KNN)
# Ensemble methods used:

- Voting Regressor
- Stacking Regressor
# The models are evaluated using metrics such as:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
## Visualizations
# The project generates visualizations to analyze:

- Distribution of injuries
- Age distribution of injured players
- Recovery times based on age and club budget
- Performance metrics for regression models

#
All figures are saved in the figures directory.