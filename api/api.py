from flask import Flask, request, jsonify
from flask_cors import CORS
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
CORS(app)

# Load the pre-trained logistic re model model using pickle
with open('../models/logistic_regression_model-09-10-2024-08-34-45-00.pkl', 'rb') as f:
    lr_model = pickle.load(f)
    

from script.default_estimator_and_WoE_binning import Estimator
lr_processor=Estimator()

def preprocess_input_rf(df):
    rfms_scores=lr_processor.calculate_rfms(df)
    rfms_labeled = lr_processor.assign_good_bad_labels(rfms_scores)
    merged_data=lr_processor.merge_dataframes(df,rfms_labeled)
    y_var,breaks=lr_processor.woe_num(merged_data,'RiskLabel')
    break_list=lr_processor.bin_catagorical_features(merged_data)
    breaks.update(break_list)
    bins_adj = sc.woebin(merged_data, 'RiskLabel', breaks_list= breaks, positive = 'bad|0')
    merged_data_woe = sc.woebin_ply(merged_data, bins_adj)
    data_final = merged_data.merge(merged_data_woe, how = 'left', left_index=True, right_index=True)
    data_final = data_final.drop(columns = 'RiskLabel_y').rename(columns={'RiskLabel_x':'vd'})
    categorical_columns = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    data_final = data_final.drop(columns = categorical_columns)
    info_values_df=lr_processor.calculate_information_value(data_final)
    data_final=lr_processor.filter_columns_by_info_value(data_final,info_values_df, threshold=0.02)
    X_data=data_final.loc[:,data_final.columns != 'vd']
    
    return X_data

    
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
