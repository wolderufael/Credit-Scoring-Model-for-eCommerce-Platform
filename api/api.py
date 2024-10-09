from flask import Flask, request, jsonify
import pandas as pd
import pickle
import datetime
from sklearn.preprocessing import  MinMaxScaler
from scipy.stats import zscore
import numpy as np
import sys
import os
import scorecardpy as sc

# Add the project directory to sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.abspath(os.path.join(current_directory, '..'))  # Adjust '..' based on project structure
sys.path.append(project_directory)

sys.path.append(os.path.abspath('../models'))

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained logistic re model model using pickle
with open('../models/logistic_regression_model-09-10-2024-00-45-34-00.pkl', 'rb') as f:
    lr_model = pickle.load(f)
    

from script.default_estimator_and_WoE_binning import Estimator
lr_processor=Estimator()

def preprocess_input_rf(df):
    # Ensure the index is set as a column named 'Unnamed: 0'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Unnamed: 0'}, inplace=True)
    rfms_scores=lr_processor.calculate_rfms(df)
    rfms_labeled = lr_processor.assign_good_bad_labels(rfms_scores)
    merged_data=lr_processor.merge_dataframes(df,rfms_labeled)
    train, test=lr_processor.split_data(merged_data)
    y_var,breaks=lr_processor.woe_num(train,'RiskLabel')
    break_list=lr_processor.bin_catagorical_features(train)
    breaks.update(break_list)
    bins_adj = sc.woebin(merged_data, 'RiskLabel', breaks_list= breaks, positive = 'bad|0')
    train_final,test_final=lr_processor.converting_into_woe_values(train,test,bins_adj)
    info_values_df=lr_processor.calculate_information_value(train_final)
    train_final=lr_processor.filter_columns_by_info_value(train_final,info_values_df, threshold=0.02)
    test_final=lr_processor.filter_columns_by_info_value(test_final,info_values_df, threshold=0.02)
    lr,lr_model,train_pred,test_pred,y_train,y_test,X_test=lr_processor.predict_risk_logistic_regressor(train_final,test_final)

    return X_test

    
# Define logistic regression prediction endpoint for CSV file input
@app.route('/predict_logistic_regression', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        input_data = pd.read_csv(file)

        processed_data = preprocess_input_rf(input_data)

        predictions = lr_model.predict(processed_data)

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Define a health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running', 'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
