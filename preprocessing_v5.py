import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing as pp
from sklearn.naive_bayes import GaussianNB

# Correlation analysis from step3(i.e preprocessing_v3 showed that MMRCurrentRetailCleanPrice, MMRCurrentAuctionCleanPrice,
# 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitonRetailCleanPrice' are highly correlated with their Average counterparts
# Hence we remove those columns from the data set and do the prediction.

attribute_types = {
    "Nominal": ["RefId", "Auction", "VehYear", "Make", "Model", "Trim", "SubModel", "Color", "Transmission", "WheelTypeID", "WheelType", "Nationality", "TopThreeAmericanName",  "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1", "VNST", "PurchDate"],
    "Binary": ["IsBadBuy", "IsOnlineSale"],
    "Numeric": ["VehicleAge", "VehOdo", "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionRetailAveragePrice", "MMRCurrentAuctionAveragePrice", "MMRCurrentRetailAveragePrice", "VehBCost", "WarrantyCost"],
    "Ordinal": ["Size"]
}

training_data = pd.read_csv("training.csv")
test_data = pd.read_csv("test.csv")

le = pp.LabelEncoder()

columns_to_select = []
for key in attribute_types:
    if key == "Nominal" or key == "Binary" or key == "Ordinal":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            if column != "RefId" and column != "IsBadBuy":
                mode_value = stats.mode(training_data[column], axis=None)
                training_data[column].fillna(mode_value[0], inplace=True)
                le.fit(training_data[column])
                training_data[column] = le.transform(training_data[column])
                columns_to_select.append(column)
    elif key == "Numeric":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            median_value = np.median(training_data[column])
            training_data[column].fillna(median_value, inplace=True)
            columns_to_select.append(column)
    
for key in attribute_types:
    if key == "Nominal" or key == "Binary" or key == "Ordinal":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            if column != "RefId" and column != "IsBadBuy":
                mode_value = stats.mode(test_data[column], axis=None)
                test_data[column].fillna(mode_value[0], inplace=True)
                le.fit(test_data[column])
                test_data[column] = le.transform(test_data[column])    
    elif key == "Numeric":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            median_value = np.median(test_data[column])
            test_data[column].fillna(median_value, inplace=True)

training_data_new = training_data[columns_to_select]
test_data_new = test_data[columns_to_select]

gnb = GaussianNB()
model = gnb.fit(training_data_new, training_data.IsBadBuy)
result = model.predict(test_data_new)

result_df = pd.DataFrame({'RefId': test_data['RefId'], 'IsBadBuy': result})
result_df.to_csv("preprocessing_result_v5.csv", index=False, columns=["RefId", "IsBadBuy"], sep=",")