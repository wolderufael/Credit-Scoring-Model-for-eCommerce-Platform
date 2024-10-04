import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def data_overview(self,df):
        num_rows = df.shape[0]
        num_columns = df.shape[1]
        data_types = df.dtypes

        print(f"Number of rows:{num_rows}")
        print(f"Number of columns:{num_columns}")
        print(f"Data types of each column:\n{data_types}")
        
    def summarize_dataset(self,df):
        # Summary statistics for 'Amount' features
        summary_stats = {
            'Mean': df['Amount'].mean(),
            'Median': df['Amount'].median(),
            'Mode': df['Amount'].mode().iloc[0],  # Taking the first mode in case of multiple modes
            'Standard Deviation': df['Amount'].std(),
            'Variance': df['Amount'].var(),
            'Range': df['Amount'].max() - df['Amount'].min(),
            'IQR': df['Amount'].quantile(0.75) - df['Amount'].quantile(0.25),
            'Skewness': df['Amount'].skew(),
            'Kurtosis': df['Amount'].kurtosis()
        }
        
        # Convert summary stats to DataFrame with an index
        summary_df = pd.DataFrame([summary_stats], index=['Summary of Amount'])
        
        return summary_df
    
    def plot_numerical_distributions(self,df):
        # Select numerical columns only
        numerical_columns=['Amount','Value']
        
        # Set style for seaborn plots
        sns.set_theme(style="whitegrid")
        
        # Create histograms and boxplots for each numerical feature
        for col in numerical_columns:
            plt.figure(figsize=(12, 6))
            
            # Plot Histogram 
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=30, color='blue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            
            # Plot Boxplot for outliers
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col], color='orange')
            plt.title(f'Boxplot of {col}')
            
            plt.tight_layout()
            plt.show()

    def plot_categorical_distributions(self,df):
        # Select categorical columns only
        categorical_columns=['ProviderId','ProductId','ProductCategory','ChannelId','PricingStrategy','FraudResult']
        
        # Create bar plots for each categorical feature
        for col in categorical_columns:
            plt.figure(figsize=(10, 5))
            
            # Plot a bar chart for the frequency of each category
            sns.countplot(x=df[col], order=df[col].value_counts().index, hue=df[col], palette="Set2", legend=False)

            # sns.countplot(x=df[col], order=df[col].value_counts().index, palette="Set2")
            
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

    def check_missing_value(self,df):
        #check missing value
        if(df.isnull().sum().sum()):
            print(f"The number of missing values:{df.isnull().sum().sum()}")
            # print("There no null value in the data")
        else:
            # print(f"The number of null values:{df.isnull().sum().sum()}")
            print("There is no missing value in the data")

    def plot_outliers_boxplots(self,df):
        # Select numerical columns only
        numerical_columns = ['Amount', 'Value']
        
        # Create box plots for each numerical feature to detect outliers
        for col in numerical_columns:
            fig = plt.figure(figsize=(10, 7))
            
            # Create the box plot
            box = plt.boxplot(df[col], patch_artist=True, boxprops=dict(facecolor='lightgreen'), 
                            medianprops=dict(color='brown'), 
                            whiskerprops=dict(color='black', linewidth=1.5), 
                            capprops=dict(color='black', linewidth=1.5))
            
            # Set title and labels
            plt.title(f'Boxplot of {col} (Outlier Detection)', fontsize=16)
            plt.xlabel(col, fontsize=14)
            
            # Set y-axis limits
            if(col=='Value'):
                plt.ylim(df[col].min(), df[col].max() * 0.002)  # Adjust as necessary to see the box clearly
            elif(col=='Amount'):
                plt.ylim(df[col].min()* 0.01,  df[col].max()* 0.002)  # Adjust as necessary to see the box clearly
            
            # Show the plot
            plt.grid()
            plt.tight_layout()
            plt.show()