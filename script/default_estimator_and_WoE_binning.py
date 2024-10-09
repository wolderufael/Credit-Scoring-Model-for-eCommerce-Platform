import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import pickle
from datetime import datetime
import os
import sidetable
# %matplotlib inline
import seaborn as sns
import scorecardpy as sc
from monotonic_binning.monotonic_woe_binning import Binning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report, roc_curve


class Estimator:
    def calculate_rfms(self,df):
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Get the last date in the data for recency calculation
        current_date = df['TransactionStartTime'].max()

        # Recency: days since last transaction
        recency = df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda x: (current_date - x.max()).days)
        )

        # Frequency: number of transactions
        frequency = df.groupby('CustomerId').agg(Frequency=('TransactionId', 'count'))

        # Monetary: total transaction amount
        monetary = df.groupby('CustomerId').agg(Monetary=('Amount', 'sum'))

        # Stability: standard deviation of transaction amounts (measure of consistency)
        stability = df.groupby('CustomerId').agg(Stability=('Amount', 'std')).fillna(0)

        # Combine RFMS features into a single DataFrame
        rfms = recency.join([frequency, monetary, stability])
        
        return rfms
    
    def visualize_rfms(self, rfms):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create scatter plot with Stability as the color
        sc = ax.scatter(rfms['Recency'], rfms['Frequency'], rfms['Monetary'], 
                        c=rfms['Stability'], cmap='coolwarm', s=50)

        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        plt.title('RFMS Space - Recency, Frequency, Monetary, Stability')

        # Add colorbar and link it to the scatter plot
        plt.colorbar(sc, label='Stability')

        plt.show()
        
    def assign_good_bad_labels(self,rfms):
        # Calculate quartiles for each feature
        rfms['RiskScore'] = (rfms['Recency'].rank(ascending=True) + 
                            rfms['Frequency'].rank(ascending=False) + 
                            rfms['Monetary'].rank(ascending=False) + 
                            rfms['Stability'].rank(ascending=False))
        
        # Calculate the first quartile (Q1)
        first_quartile = rfms['RiskScore'].quantile(0.25)
        # Assign RiskLabel based on whether RiskScore is above or below Q1
        # rfms['RiskLabel'] = np.where(rfms['RiskScore'] > first_quartile, 1, 0)

        # Classify into Good (low risk) and Bad (high risk)
        rfms['RiskLabel'] = np.where(rfms['RiskScore'] > rfms['RiskScore'].median(), 1, 0)

        return rfms

    def merge_dataframes(self,df1, df2):
        columns_to_drop = ['TransactionId','TransactionStartTime', 'BatchId', 'AccountId', 'SubscriptionId', 'CurrencyCode', 'CountryCode', 'Value']
        df1 = df1.drop(columns=columns_to_drop)
        
        # Merge the two DataFrames on 'CustomerId'
        merged_df = pd.merge(df1, df2, on='CustomerId', how='inner')  
        merged_df.drop(columns='CustomerId',inplace=True)
        # Define the mapping
        pricing_strategy_map = {0: 'strat0', 1: 'strat1', 2: 'strat2', 4: 'strat4'}

        # # Replace the values in the PricingStrategy column
        merged_df['PricingStrategy'] = merged_df['PricingStrategy'].replace(pricing_strategy_map)
        
        return merged_df
    
    def split_data(self,df):
        train, test = sc.split_df(df, 'RiskLabel', ratio = 0.7, seed = 999).values()
        
        return train,test
    
    def woe_num(self,train,y):  
        # WoE (Weight of Evidence) Transformation
        x = train.drop(['RiskLabel', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy'], axis = 1).columns
        y_var = train['RiskLabel']   
                        
        bin_object = Binning(y, n_threshold = 50, y_threshold = 10, p_threshold = 0.35, sign=False)
        global breaks
        breaks = {}
        for i in x:
            bin_object.fit(train[[y, i]])
            breaks[i] = (bin_object.bins[1:-1].tolist())
        return y_var,breaks
    
    def bin_catagorical_features(self,train):
        categorical_columns = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        bins = sc.woebin(train, y = 'RiskLabel', x = categorical_columns)
                        #  , save_breaks_list = 'cat_breaks')
        # Extract the breaks from the resulting 'bins' object
        breaks_list = {col: bins[col]['breaks'].tolist() for col in bins.keys()}
        
        return breaks_list
        
    def plot_woe_iv(self,df,breaks):
        bins_adj = sc.woebin(df, 'RiskLabel', breaks_list= breaks, positive = 'bad|0')
        sc.woebin_plot(bins_adj)
        
        return bins_adj
        
    def converting_into_woe_values(self,train,test,bins_adj):
        train_woe = sc.woebin_ply(train, bins_adj)
        test_woe = sc.woebin_ply(test, bins_adj)
        
        # Merge by index
        train_final = train.merge(train_woe, how = 'left', left_index=True, right_index=True)
        test_final = test.merge(test_woe, how = 'left', left_index=True, right_index=True)
        
        train_final = train_final.drop(columns = 'RiskLabel_y').rename(columns={'RiskLabel_x':'vd'})
        test_final = test_final.drop(columns = 'RiskLabel_y').rename(columns={'RiskLabel_x':'vd'})
        
        # # Drop transformed variables
        categorical_columns = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        train_final = train_final.drop(columns = categorical_columns)
        test_final = test_final.drop(columns = categorical_columns)
        
        return train_final,test_final
    
    def calculate_information_value(self,train_final):
        iv=sc.iv(train_final, y = 'vd')
        
        return iv

    def filter_columns_by_info_value(self,df,info_values_df, threshold=0.02):
        # Get the variables (columns) that meet the info_value threshold
        # columns_to_keep = info_values_df[info_values_df['info_value'] >= threshold]['variable'].tolist()
        columns_to_keep=['Stability','Monetary','RiskScore','Frequency','Recency','Recency_woe','Amount','Frequency_woe','ProviderId_woe','ProductId_woe','Amount_woe','PricingStrategy_woe']
    
        # Ensure 'vd' is included in the final DataFrame
        if 'vd' in df.columns:
            columns_to_keep.append('vd')
        
        # Filter the train_final DataFrame to only include these columns
        filtered_df = df[columns_to_keep]
        
        # Move 'vd' to the last column if it exists
        if 'vd' in filtered_df.columns:
            vd_column = filtered_df.pop('vd')  # Remove 'vd' column
            filtered_df['vd'] = vd_column      # Add 'vd' back as the last column
    
        return filtered_df
    
    def filter_variables(self,train_final):
        filtered=sc.var_filter(train_final, y = 'vd')
        
        return filtered
    
    def predict_risk_logistic_regressor(self,train_final,test_final):
        y_train = train_final.loc[:,'vd']
        X_train = train_final.loc[:,train_final.columns != 'vd']
        y_test = test_final.loc[:,'vd']
        X_test = test_final.loc[:,train_final.columns != 'vd']
        
        lr = LogisticRegression(penalty='l1', C=0.9, solver='liblinear')
        lr_model=lr.fit(X_train, y_train)
        
        # Print the coefficients
        print(lr.coef_)
        
        # # predicted proability
        train_pred = lr.predict_proba(X_train)[:,1]
        test_pred = lr.predict_proba(X_test)[:,1]
        
        return lr,lr_model,train_pred,test_pred,y_train,y_test,X_test
    
    def predict_risk_random_forrest(self,train_final,test_final):
        from sklearn.model_selection import GridSearchCV
        
        y_train = train_final.loc[:,'vd']
        X_train = train_final.loc[:,train_final.columns != 'vd']
        y_test = test_final.loc[:,'vd']
        X_test = test_final.loc[:,train_final.columns != 'vd']
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, max_features='sqrt', random_state=42)

        # rf_model = rf.fit(X_train, y_train)

        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2']
        }

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Use the best model found by GridSearchCV
        best_rf_model = grid_search.best_estimator_

        # Print feature importances from the best model
        print("Feature Importances:", best_rf_model.feature_importances_)

        # Predicted probabilities using the best model
        train_pred = best_rf_model.predict_proba(X_train)[:, 1]
        test_pred = best_rf_model.predict_proba(X_test)[:, 1]
        # # Print feature importances
        # print("Feature Importances:", rf.feature_importances_)
        
        # # # predicted proability
        # train_pred = rf.predict_proba(X_train)[:,1]
        # test_pred = rf.predict_proba(X_test)[:,1]
        
        return rf,best_rf_model,train_pred,test_pred,y_train,y_test,X_test
    
    def performance_ks_roc(self,train_pred,test_pred,y_train,y_test):
        train_perf = sc.perf_eva(y_train, train_pred, title = "train")
        test_perf = sc.perf_eva(y_test, test_pred, title = "test")
        
        return train_perf,test_perf
    
    def evaluation(self,lr,X_test,y_test):
        predictions = lr.predict(X_test)
        
        print('Accuracy')
        print(accuracy_score(y_test, predictions))
        print('AUC Score')
        print(roc_auc_score(y_test, predictions))
        
        print(classification_report(y_test,predictions))
        
        return predictions
    
    def confusion_matrix(self,y_test,predictions):
        conf_log2 = confusion_matrix(y_test,predictions)
        sns.heatmap(data=conf_log2, annot=True, linewidth=0.7, linecolor='k', fmt='.0f', cmap='magma')
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.title('Confusion Matrix - Logistic Regression');
        
    def save_model_with_timestamp(self,model,name, folder_path='models/'):
        logging.info('Serializes and saves a trained model with a timestamp.')
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Generate timestamp in format dd-mm-yyyy-HH-MM-SS-00
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
        
        # Create a filename with the timestamp
        filename = f'{folder_path}{name}-{timestamp}.pkl'
        
        # Save the model using pickle
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        
        print(f"Model saved as {filename}")
        return filename