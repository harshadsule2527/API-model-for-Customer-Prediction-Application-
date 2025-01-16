 # Customer Behavior Prediction and Product Recommendation System

This project is a Flask-based web application that predicts customer behavior, such as the likelihood of product returns and repeat purchases, using machine learning models. It also provides product recommendations using collaborative filtering.

## Features

1. **Customer Behavior Prediction**:
   - Predicts the probability of a customer returning a product.
   - Predicts the likelihood of a customer making a repeat purchase.

2. **Product Recommendation**:
   - Recommends products to customers based on collaborative filtering.

3. **REST API**:
   - Exposes endpoints to predict customer behavior via JSON payloads.

4. **Web Interface**:
   - Allows users to input customer data via a web form and view predictions.

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the dataset (`customer_data.csv`) in the project directory.

### Running the Application

1. Train and save the models:
   ```bash
   python app.py
   ```

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Access the web application at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Project Structure

```plaintext
.
├── app.py                  # Main application code
├── customer_data.csv       # Dataset for training and predictions
├── requirements.txt        # Required Python packages
├── templates/
│   └── prediction_form.html # HTML template for the web interface
├── return_model.joblib     # Trained model for return likelihood (generated after training)
├── repeat_model.joblib     # Trained model for repeat purchase likelihood (generated after training)
├── preprocessor.joblib     # Data preprocessor (generated after training)
├── svd_model.joblib        # Truncated SVD model for recommendations (generated after training)
├── latent_matrix.joblib    # Latent matrix for recommendations (generated after training)
├── user_item_matrix.pkl    # User-item matrix for recommendations (generated after training)
```

## API Endpoints

### Predict Customer Behavior

- **Endpoint**: `/predict`
- **Method**: `POST`
- **Request Body** (JSON):
  ```json
  {
      "customer_data": {
          "feature1": value1,
          "feature2": value2,
          ...
      }
  }
  ```
- **Response**:
  ```json
  {
      "return_likelihood": "0.85",
      "repeat_purchase_likelihood": "0.75"
  }
  ```

### Web Form

Access the form via the root endpoint (`/`). Enter customer details to get predictions.

## Data Preparation

- The dataset should contain numerical and categorical features.
- Ensure the following columns exist for product recommendations:
  - `customer_id`
  - `category`
  - `total_spend`

## Model Details

1. **Return Likelihood Prediction**:
   - Random Forest Classifier.
   - Trained on processed numerical and categorical features.

2. **Repeat Purchase Prediction**:
   - Random Forest Classifier.
   - Trained similarly to the return model.

3. **Product Recommendations**:
   - Collaborative filtering using Truncated SVD.
   - User-item matrix constructed from `customer_id`, `category`, and `total_spend`.

## Evaluation Metrics

The models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Issues and Warnings

- Ensure the dataset has sufficient data for both behavior prediction and recommendations.
- If models always predict a single class, inspect the data distribution.

## Contributing

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)

