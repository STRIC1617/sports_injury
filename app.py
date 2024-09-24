from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models
models = {
    "K Nearest Neighbour Regression": joblib.load('models/k-nearest_neighbors_regression_model.pkl'),
    "Decision Tree": joblib.load('models/decision_tree_regression_model.pkl'),
    "Linear Regression": joblib.load('models/linear_regression_regression_model.pkl'),
    # Add other models as needed
}

# Load club statistics for dropdown
club_stats = pd.read_csv('dataset/player_injury_data.csv')
club_stats.columns = club_stats.columns.str.strip()  # Strip whitespace from column names
unique_clubs = club_stats['club'].unique()

# Define injury types (this should match your encoding)
injury_types = {
    "Muscle Injury": 0,
    "Ligament Injury": 1,
    "Fracture": 2,
    "Sprain": 3,
    "Tendon Injury": 4,
    # Add other types as needed
}

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        club_name = request.form['club']
        age = int(request.form['age'])
        injury_type = injury_types[request.form['injury_type']]

        # Fetch club data
        club_data = club_stats[club_stats['club'] == club_name].iloc[0]
        print(club_data)

        # Ensure 'avg_recovery_time' exists
        avg_recovery_time = club_data.get('days', None)  # Use get to avoid KeyError
        if avg_recovery_time is None:
            raise ValueError(f"'avg_recovery_time' not found for club: {club_name}")

        # Prepare input for prediction
        bmi = weight / (height ** 2)
        club_value = club_data['club_value']
        input_data = [[weight, height, club_value, age, injury_type, bmi, avg_recovery_time]]

        # Store predictions
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(input_data)
            predictions[model_name] = prediction[0]

        return render_template('index.html', predictions=predictions, clubs=unique_clubs, injury_types=injury_types)


    return render_template('index.html', predictions=None, clubs=unique_clubs, injury_types=injury_types)

if __name__ == '__main__':
    app.run(debug=True)
