import xgboost as xgb
import pickle
import config
import os
import numpy as np
from helpers import get_by_name, get_trainable, permute_together

test_train_split = 250000

# ----------------------------------- Train -----------------------------------
dataset_name = "Tests/like_previous_code"
dataset = get_by_name(dataset_name, config.WRAPPER_DICT_NAME)['training_data_builder']

trainable_data, trainable_pt, trainable_event_weight, trainable_filtered_data = get_trainable(dataset)
permute_together(trainable_data, trainable_pt, trainable_event_weight, trainable_filtered_data)

# training_data = trainable_data
# train_pt = trainable_pt
# event_weight = trainable_event_weight
training_data = trainable_data[:test_train_split]
train_pt = trainable_pt[:test_train_split]
event_weight = trainable_event_weight[:test_train_split]

xg_reg = xgb.XGBRegressor(objective = 'reg:linear', 
                        learning_rate = .1, 
                        max_depth = 5, 
                        n_estimators = 400,
                        max_bins = 1000,
                        nthread = 30)

xg_reg.fit(training_data, train_pt, sample_weight = event_weight)

model_path = os.path.join(config.RESULTS_DIRECTORY, dataset_name, config.MODEL_NAME)
with open(model_path, 'wb') as file:
    pickle.dump(xg_reg, file)

print("------------------------------ Feature Importances ---------------------------")
feature_importances = xg_reg.feature_importances_
for name, importance in zip(np.array(dataset.feature_names)[dataset.trainable_features], feature_importances):
    print(f"{name}:\t {importance}")
print("\n")
print("------------------------------------------------------------------------------")


# ----------------------------------- Predict -----------------------------------
# dataset_name = "Tests/shower_bit_wrong_distribution"
# dataset = get_by_name(dataset_name, config.WRAPPER_DICT_NAME)['training_data_builder']
# testing_data, testing_pt, _, testing_filtered_data = get_trainable(dataset)

testing_data = trainable_data[test_train_split:]
testing_filtered_data = trainable_filtered_data[test_train_split:]

predicted_pt = np.exp(xg_reg.predict(testing_data))

test_dict = {
    "predicted_pt"      : predicted_pt,
    "training_features" : np.array(dataset.feature_names)[dataset.trainable_features],
    "testing_data"      : testing_data,
    "gen_features"      : np.array(dataset.feature_names)[~dataset.trainable_features],
    "gen_data"          : testing_filtered_data[:, ~dataset.trainable_features]
}

prediction_path = os.path.join(config.RESULTS_DIRECTORY, dataset_name, config.PREDICTION_NAME)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)