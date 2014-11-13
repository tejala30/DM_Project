import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing as pp
from sklearn.naive_bayes import GaussianNB


def correlation(given_list):
    corr_dict = {}
    for col_name in given_list:
    # new_num_list = [x for x in numeric_list if x != col_name]
        for col_name1 in given_list:
            col_list = []
            if col_name != col_name1:
                s1 = pd.Series(training_data[col_name])
                s2 = pd.Series(training_data[col_name1])
                col_list.append(str(col_name))
                col_list.append(str(col_name1))
                col_list.sort()
                key = col_list[0] + " " + col_list[1]
                if s1.corr(s2) >= 0.9 and key not in corr_dict.keys():
                    corr_dict[key] = s1.corr(s2)

    return corr_dict


attribute_types = {
    "Nominal": ["RefId", "Auction", "VehYear", "Make", "Model", "Trim", "SubModel", "Color", "Transmission", "WheelTypeID", "WheelType", "Nationality", "TopThreeAmericanName",  "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1", "VNST", "PurchDate"],
    "Binary": ["IsBadBuy", "IsOnlineSale"],
    "Numeric": ["VehicleAge", "VehOdo", "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice", "VehBCost", "WarrantyCost"],
    "Ordinal": ["Size"]
}

training_data = pd.read_csv("training.csv")
test_data = pd.read_csv("test.csv")

le = pp.LabelEncoder()

for key in attribute_types:
    if key == "Nominal" or key == "Binary" or key == "Ordinal":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            if column != "RefId" and column != "IsBadBuy":
                mode_value = stats.mode(training_data[column], axis=None)
                training_data[column].fillna(mode_value[0], inplace=True)
                le.fit(training_data[column])
                training_data[column] = le.transform(training_data[column])
    elif key == "Numeric":
        for i in range(0, len(attribute_types[key])):
            column = attribute_types[key][i]
            median_value = np.median(training_data[column])
            training_data[column].fillna(median_value, inplace=True)

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

numeric_list = []
nominal_list = []
for key in attribute_types:
    if key == "Numeric":
        numeric_list = attribute_types[key]
    elif key == "Nominal":
        nominal_list = attribute_types[key]

print("Correlated Numeric Attributes")
num_corr_dict = correlation(numeric_list)
for k, v in num_corr_dict.items():
    print(str(k) + ": " + str(v))

print("\n" + "Correlated Nominal Attributes")
nom_corr_dict = correlation(nominal_list)
for k, v in nom_corr_dict.items():
    print(str(k) + ": " + str(v))

training_columns = training_data.columns[(training_data.columns != 'RefId') & (training_data.columns != 'IsBadBuy')]
test_columns = test_data.columns[test_data.columns != 'RefId']

training_data_new = training_data[training_columns]
test_data_new = test_data[test_columns]

gnb = GaussianNB()
model = gnb.fit(training_data_new, training_data.IsBadBuy)
result = model.predict(test_data_new)

result_df = pd.DataFrame({'RefId': test_data['RefId'], 'IsBadBuy': result})
result_df.to_csv("preprocessing_result_v2.csv", index=False, columns=["RefId", "IsBadBuy"], sep=",")
