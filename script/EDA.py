import pandas as pd

class EDA:
    def data_overview(self,df):
        # df.info()
        num_rows = df.shape[0]
        num_columns = df.shape[1]
        data_types = df.dtypes

        print(f"Number of rows:{num_rows}")
        print(f"Number of columns:{num_columns}")
        print(f"Data types of each column:\n{data_types}")