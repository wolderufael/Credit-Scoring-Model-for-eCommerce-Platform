import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        
        # Classify into Good (low risk) and Bad (high risk)
        rfms['RiskLabel'] = np.where(rfms['RiskScore'] > rfms['RiskScore'].median(), 1, 0)

        return rfms

    def merge_dataframes(self,df1, df2):
        # Merge the two DataFrames on 'CustomerId'
        merged_df = pd.merge(df1, df2, on='CustomerId', how='inner')  
        merged_df.drop(columns='CustomerId',inplace=True)
        
        return merged_df