import pandas as pd
from scipy import stats
from sklearn import preprocessing as pp
from sklearn.naive_bayes import GaussianNB

training_data = pd.read_csv("training.csv")
test_data = pd.read_csv("test.csv")

le = pp.LabelEncoder()

for column in training_data.columns:
    if str(training_data[column].dtype) == 'object':
        mode_value = stats.mode(training_data[column], axis=None)
        training_data[column].fillna(mode_value[0], inplace=True)
        le.fit(training_data[column])
        training_data[column] = le.transform(training_data[column])
    else:
        mode_value = stats.mode(training_data[column], axis=None)
        training_data[column].fillna(mode_value[0], inplace=True)

for column in test_data.columns:
    if str(test_data[column].dtype) == 'object':
        mode_value = stats.mode(test_data[column], axis=None)
        test_data[column].fillna(mode_value[0], inplace=True)
        le.fit(test_data[column])
        test_data[column] = le.transform(test_data[column])
    else:
        mode_value = stats.mode(training_data[column], axis=None)
        test_data[column].fillna(mode_value[0], inplace=True)

training_columns = training_data.columns[(training_data.columns != 'RefId') & (training_data.columns != 'IsBadBuy')]
test_columns = test_data.columns[test_data.columns != 'RefId']

training_data_new = training_data[training_columns]
test_data_new = test_data[test_columns]

gnb = GaussianNB()
model = gnb.fit(training_data_new, training_data.IsBadBuy)
result = model.predict(test_data_new)

result_df = pd.DataFrame({'RefId': test_data['RefId'], 'IsBadBuy': result})
result_df.to_csv("preprocessing_result_v1.csv", index=False, columns=["RefId", "IsBadBuy"], sep=",")