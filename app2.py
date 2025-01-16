import pandas as pd
import numpy as np  
import pandas as pd
import requests
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, render_template


# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the dataset:", df.columns)
    
    # Identify numerical and categorical columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # Remove target variables from features if they exist
    features_to_remove = ['customer_id', 'return_likelihood', 'repeat_purchase']
    numerical_features = [col for col in numerical_features if col not in features_to_remove]
    categorical_features = [col for col in categorical_features if col not in features_to_remove]
    
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return df, preprocessor

# Train models
def train_model(X, y, model_type='return'):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Product recommendation function using collaborative filtering
def train_recommendation_model(user_item_matrix):
    from sklearn.decomposition import TruncatedSVD
    n_components = min(10, user_item_matrix.shape[1] - 1)  # Use minimum of 10 or number of features - 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(user_item_matrix)
    return svd, latent_matrix

def recommend_products(user_id, svd, latent_matrix, user_item_matrix, n_recommendations=5):
    user_preferences = latent_matrix[user_id]
    all_items_preferences = svd.components_.T.dot(user_preferences)
    already_bought = user_item_matrix[user_id].nonzero()[1]
    recommended_items = np.argsort(all_items_preferences)[::-1]
    recommended_items = [item for item in recommended_items if item not in already_bought][:n_recommendations]
    return recommended_items

# Load and prepare data
data, preprocessor = load_and_preprocess_data('customer_data.csv')

# Check if target variables exist in the dataset
if 'return_likelihood' in data.columns and 'repeat_purchase' in data.columns:
    X = data.drop(['customer_id', 'return_likelihood', 'repeat_purchase'], axis=1, errors='ignore')
    y_return = data['return_likelihood']
    y_repeat = data['repeat_purchase']
else:
    print("Warning: Target variables 'return_likelihood' and/or 'repeat_purchase' not found in the dataset.")
    print("Using dummy target variables for demonstration purposes.")
    X = data.drop('customer_id', axis=1, errors='ignore')
    y_return = np.random.randint(0, 2, size=len(X))
    y_repeat = np.random.randint(0, 2, size=len(X))

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_return_train, y_return_test, y_repeat_train, y_repeat_test = train_test_split(
    X_processed, y_return, y_repeat, test_size=0.2, random_state=42
)

# Train models
return_model = train_model(X_train, y_return_train, 'return')
repeat_model = train_model(X_train, y_repeat_train, 'repeat')

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

print("Return Model Evaluation:", evaluate_model(return_model, X_test, y_return_test))
print("Repeat Purchase Model Evaluation:", evaluate_model(repeat_model, X_test, y_repeat_test))

# Train recommendation model
if 'customer_id' in data.columns and 'category' in data.columns and 'total_spend' in data.columns:
    user_item_matrix = pd.pivot_table(data, values='total_spend', index='customer_id', columns='category', fill_value=0)
    if user_item_matrix.shape[1] > 1:  # Ensure we have at least 2 columns (features)
        svd, latent_matrix = train_recommendation_model(user_item_matrix)
    else:
        print("Warning: Not enough categories for recommendation. Recommendation system will not be available.")
        svd, latent_matrix, user_item_matrix = None, None, None
else:
    print("Warning: Required columns for recommendation not found. Recommendation system will not be available.")
    svd, latent_matrix, user_item_matrix = None, None, None

# After training the models
for model_name, model, X, y in [("Return", return_model, X_test, y_return_test), 
                                ("Repeat Purchase", repeat_model, X_test, y_repeat_test)]:
    evaluation = evaluate_model(model, X, y)
    print(f"{model_name} Model Evaluation:", evaluation)
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"{model_name} Target Distribution:", dict(zip(unique, counts)))
    
    # If the model always predicts the same class, print a warning
    if len(np.unique(model.predict(X))) == 1:
        print(f"Warning: {model_name} model is predicting only one class. This might indicate a problem with the data or model.")

# Save models
joblib.dump(return_model, 'return_model.joblib')
joblib.dump(repeat_model, 'repeat_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(svd, 'svd_model.joblib')
joblib.dump(latent_matrix, 'latent_matrix.joblib')
user_item_matrix.to_pickle('user_item_matrix.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Handle both form data and JSON input
            if request.is_json:
                data = request.json
                customer_data = pd.DataFrame([data['customer_data']])
            else:
                customer_data = pd.DataFrame([request.form.to_dict()])
            
            # Ensure customer_data only contains columns that the model was trained on
            valid_columns = preprocessor.feature_names_in_
            customer_data = customer_data[valid_columns]
            
            # Preprocess the input data
            customer_data_processed = preprocessor.transform(customer_data)
            
            return_likelihood = return_model.predict_proba(customer_data_processed)[:, 1][0]
            repeat_purchase_likelihood = repeat_model.predict_proba(customer_data_processed)[:, 1][0]
            
            prediction = {
                'return_likelihood': f"{return_likelihood:.2f}",
                'repeat_purchase_likelihood': f"{repeat_purchase_likelihood:.2f}"
            }
            
            # If it's an API call, return JSON
            if request.is_json:
                return jsonify(prediction), 200
            
        except Exception as e:
            # If it's an API call, return JSON error
            if request.is_json:
                return jsonify({'error': str(e)}), 400
            # For web form, set prediction to error message
            prediction = {'error': str(e)}
    
    # Render the HTML template for both GET and POST (form submission)
    return render_template('prediction_form.html', prediction=prediction)

if __name__ == '__main__':
    # Load models
    return_model = joblib.load('return_model.joblib')
    repeat_model = joblib.load('repeat_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    
    app.run(debug=True)