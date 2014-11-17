import pandas as pd
import numpy as np
from numpy import nan as NA
from scipy import stats
from sklearn import preprocessing as pp
from sklearn.naive_bayes import GaussianNB
import math

attribute_types = {
    "Nominal": ["Auction", "VehYear", "Make", "Model", "Trim", "SubModel", "Color", "Transmission",
                "WheelTypeID", "WheelType", "Nationality", "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", "BYRNO",
                "VNZIP1", "VNST", "PurchDate"],
    "Binary": ["IsOnlineSale"],
    "Numeric": ["VehicleAge", "VehOdo", "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice",
                "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice",
                "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice", "VehBCost",
                "WarrantyCost"],
    "Ordinal": ["Size"]
}

training_data = pd.read_csv("training.csv")
test_data = pd.read_csv("test.csv")

columns_with_all_data = []
columns_with_missing_data = []

for column in training_data.columns:
    if training_data[column].count() == training_data[column].shape[0]:
        if column != 'IsBadBuy':
            columns_with_all_data.append(column)
    else:
        columns_with_missing_data.append(column)

# for column in columns_with_missing_data:
column = "PRIMEUNIT"
columns_to_pick = [column]
columns_to_pick.extend(columns_with_all_data)
new_tr_df = training_data[columns_to_pick]
# index = new_tr_df[column].index[new_tr_df[column].apply(np.isnan)]


training_data_for_column = new_tr_df[new_tr_df[column].notnull()]
test_data_for_column = new_tr_df[new_tr_df[column].isnull()]

test_data_for_column.drop('PRIMEUNIT', axis=1, inplace=True)
print training_data_for_column.head(20)
# print test_data_for_column.head(20)
# test_data_for_column = training_data[training_data[column] is None]


le = pp.LabelEncoder()

for key in attribute_types:
    if key == "Nominal" or key == "Binary" or key == "Ordinal":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            if column in columns_to_pick:
                if column != "RefId":
                    mode_value = stats.mode(training_data_for_column[column], axis=None)
                    training_data_for_column[column].fillna(mode_value[0], inplace=True)
                    if column == 'PRIMEUNIT':
                        print str(column) + " : " + str(mode_value[0])
                        le.inverse_transform([1, 2])
                    le.fit(training_data_for_column[column])
                    training_data_for_column[column] = le.transform(training_data_for_column[column])
    elif key == "Numeric":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            if column in columns_to_pick:
                median_value = np.median(training_data_for_column[column])
                training_data_for_column[column].fillna(median_value, inplace=True)

for key in attribute_types:
    if key == "Nominal" or key == "Binary" or key == "Ordinal":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            if column != "RefId" and column != "IsBadBuy" and column in columns_with_all_data:
                mode_value = stats.mode(test_data_for_column[column], axis=None)
                test_data_for_column[column].fillna(mode_value[0], inplace=True)
                le.fit(test_data_for_column[column])
                test_data_for_column[column] = le.transform(test_data_for_column[column])
    elif key == "Numeric":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            if column in columns_to_pick:
                median_value = np.median(test_data_for_column[column])
                test_data_for_column[column].fillna(median_value, inplace=True)
# #
# # training_columns = training_data.columns[
# #     (training_data.columns != 'RefId') & (training_data.columns != 'IsBadBuy') & (training_data.columns != 'PRIMEUNIT')]
# test_columns = test_data.columns[test_data_for_column.columns != 'PRIMEUNIT' & (test_data.columns != 'PRIMEUNIT')]
#

col1 = []
col1.extend(columns_to_pick)
col1.remove('RefId')

training_data_new = training_data_for_column[col1]
test_data_new = test_data_for_column[columns_with_all_data]
#
gnb = GaussianNB()
model = gnb.fit(training_data_new, training_data_new.PRIMEUNIT)
result = model.predict(test_data_new)
#
result_df = pd.DataFrame({'RefId': test_data_new['RefId'], 'PRIMEUNIT': result})
result_df.to_csv("preprocessing_result_v4.csv", index=False, columns=["RefId", "PRIMEUNIT"], sep=",")