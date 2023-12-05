Fraud Detection Project
Overview

This project focuses on fraud detection in insurance claims using data analysis and a machine learning model. The project includes Exploratory Data Analysis (EDA) to understand the dataset and visualize important patterns, followed by the development of a RandomForestClassifier for fraud detection.
Files:

    [insurance_claims.csv: The dataset containing information about insurance claims](https://github.com/samppaalek/fraud_detection/blob/main/insurance_claims.csv)
    fraud_detection_model.joblib: The saved RandomForestClassifier model after training.

Project Structure

    1. EDA and Data Visualization:
        Loaded the dataset and performed initial data exploration.
        Visualized the distribution of claim amounts and created a heatmap of the correlation matrix.
        Encoded categorical variables and analyzed the correlation matrix with the encoded fraud_reported column.

    2. Model Development:
        Divided the data into features (X) and the target variable (y).
        Conducted one-hot encoding for categorical variables and split the data into training and testing sets.
        Normalized numerical features using StandardScaler.
        Chose RandomForestClassifier as the model and trained it on the training set.
        Evaluated the model's performance using classification report, confusion matrix, and accuracy score.

    3. K-Fold Cross Validation:
        Utilized StratifiedKFold for cross-validation to assess model robustness.
        Displayed cross-validation scores and mean accuracy.

    4. Hyperparameter Tuning:
        Performed GridSearchCV to find the best hyperparameters for the RandomForestClassifier.
        Displayed the best hyperparameters obtained from the search.

    5. Model Deployment:
        Saved the trained model using joblib for future use.
        Loaded the saved model for predictions.

    6. Model Performance Visualization:
        Visualized the confusion matrix as a heatmap to assess model performance.

Dependencies

    pandas
    matplotlib
    seaborn
    scikit-learn
    joblib

How to Run

    Install the required dependencies: pip install -r requirements.txt.
    Run the Jupyter Notebook or script containing the project code.
