import pandas as pd


class Data:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_data(self):
        # binary encoding categorical columns
        self.data[' education'] = self.data[' education'].map({' Graduate': 1, ' Not Graduate': 0})
        self.data[' self_employed'] = self.data[' self_employed'].map({' Yes': 1, ' No': 0})
        self.data[' loan_status'] = self.data[' loan_status'].map({' Approved': 1, ' Rejected': 0})

        # check if there exists any columns with null values
        if not self.data.isnull().values.any():
            print("No null self.data")
            return

        cols_with_null = self.data.columns[self.data.isnull().any()].to_list()
        threshold = 0.5 * self.data.shape[0]
        for col in cols_with_null:
            # if over 50% of values are null, drop the column
            if self.data[col].isnull().sum() > threshold:
                self.data.drop(col, axis=1)

            # if less than 50% are null, replace them with the mean of the column
            self.data[col].fillna(value=self.data[col].mean(), inplace=True)

    def transform(self):
        # separating all the binary columns to avoid normalizing them
        columns_to_add_back = self.data[[' education', ' self_employed', ' loan_status']]
        self.data.drop(columns=columns_to_add_back, axis=1, inplace=True)

        # normalizing the data
        numerical_cols = self.data.select_dtypes(include=["number"]).columns
        mean = self.data[numerical_cols].mean()
        std = self.data[numerical_cols].std().replace(0, 1)

        self.data[numerical_cols] = (self.data[numerical_cols] - mean) / std

        # putting back the data table together
        self.data = pd.concat([self.data, columns_to_add_back], axis=1)

    def get_data(self):
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        split_index = int(len(self.data) * 0.7)     # 70% kept for training

        training_set = self.data[:split_index]
        testing_set = self.data[split_index:]

        y_train = training_set[" loan_status"]
        x_train = training_set.drop(" loan_status", axis=1)
        y_test = testing_set[" loan_status"]
        x_test = testing_set.drop(" loan_status", axis=1)

        return x_train, y_train, x_test, y_test

