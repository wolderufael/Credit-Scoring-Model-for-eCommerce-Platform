import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureEngineering:
    def create_aggregate_features(self,df):
        # Group data by CustomerId and apply aggregation functions
        aggregate_df = df.groupby('CustomerId').agg(
            total_transaction_amount=('Amount', 'sum'),            
            average_transaction_amount=('Amount', 'mean'),         
            transaction_count=('TransactionId', 'count'),         
            std_transaction_amount=('Amount', 'std')              
        ).reset_index()  

        # Fill NaN values in std_transaction_amount (e.g., for customers with only one transaction)
        aggregate_df.fillna({'std_transaction_amount':0},inplace=True)

        return aggregate_df
    
    def extract_transaction_features(self,df):
        # Convert TransactionStartTime to datetime format if not already done
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Extract hour of the transaction
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        
        # Extract day of the month when the transaction occurred
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        
        # Extract month when the transaction occurred
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        
        # Extract year when the transaction occurred
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        return df
    
    def encode_categorical_variables(self,df):
        # Define the list of categorical columns to encode
        categorical_columns = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        
        # Apply One-Hot Encoding to the specified categorical columns
        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # Find the newly created one-hot encoded columns
        new_columns = [col for col in df_encoded.columns if any(cat_col in col for cat_col in categorical_columns)]
        
        # Convert only the one-hot encoded columns to 0 and 1
        df_encoded[new_columns] = df_encoded[new_columns].astype(int)
        
        return df_encoded
    
    def normalize_features(self,df):
        numerical_columns=['Amount','Value']
        scaler = MinMaxScaler()  # Initialize MinMaxScaler for normalization
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        
        return df
    
    def save_to_csv(self,df,file_path):
        # Drop the specified columns
        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CurrencyCode', 'CountryCode', 'Value']
        df = df.drop(columns=columns_to_drop)
        
        # Save the resulting DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        
        print('Data set saved succusfully!')