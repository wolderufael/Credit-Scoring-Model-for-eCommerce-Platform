import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import sidetable
# %matplotlib inline
import seaborn as sns
import scorecardpy as sc
from monotonic_binning.monotonic_woe_binning import Binning
from sklearn.linear_model import LogisticRegression
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
        bins = sc.woebin(train, y = 'RiskLabel', x = categorical_columns, save_breaks_list = 'cat_breaks')
        
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
        
        # # predicted proability
        train_pred = lr.predict_proba(X_train)[:,1]
        test_pred = lr.predict_proba(X_test)[:,1]
        
        return lr,lr_model,train_pred,test_pred,y_train,y_test,X_test
    
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