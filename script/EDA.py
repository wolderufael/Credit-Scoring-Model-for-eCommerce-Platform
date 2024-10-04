import pandas as pd
import numpy as np
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
        # Select numerical columns only
        numerical_columns=['Amount','Value']
        
        # Initialize a list to hold summary statistics for each column
        summary_list = []
    
        for col in numerical_columns:
            summary_stats = {
                'Mean': df[col].mean(),
                'Median': df[col].median(),
                'Mode': df[col].mode().iloc[0],  # Taking the first mode in case of multiple modes
                'Standard Deviation': df[col].std(),
                'Variance': df[col].var(),
                'Range': df[col].max() - df[col].min(),
                'IQR': df[col].quantile(0.75) - df[col].quantile(0.25),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurtosis()
            }
            
            # Append the summary statistics for the current column to the list
            summary_list.append(summary_stats)
        
        # Convert summary stats list to DataFrame with appropriate index
        summary_df = pd.DataFrame(summary_list, index=numerical_columns)
        
        return summary_df
    
    def plot_numerical_distributions(self,df):
        # Select numerical columns only
        numerical_columns=['Amount','Value']
        
        # Create histograms and boxplots for each numerical feature
        for col in numerical_columns:
            plt.figure(figsize=(10, 7))
            plt.subplot(1, 2, 1)
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            # Drop NaN values for the selected column
            data = df[col].dropna()

            # Create a histogram to get frequency and bin edges
            frequency, bin_edges = np.histogram(data, bins=10)

            # Get the bin centers
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Plot the line graph for frequency distribution
            
            plt.plot(bin_centers, frequency, color='green', marker='o', linestyle='-', linewidth=2, markersize=5)

            # Adding titles and labels
            plt.title(f'Frequency Polygon (Line Graph) of {col} Distribution', fontsize=16)
            plt.xlabel(col, fontsize=14)
            plt.ylabel('Frequency', fontsize=14)

            # Set x-axis limits to the minimum and maximum values of the column
            plt.xlim(data.min(), data.max())

            # Display the grid
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Create the box plot
            plt.subplot(1, 2, 2)
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