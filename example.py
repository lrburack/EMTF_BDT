import xgboost as xgb
import pickle
import config
import os
import numpy as np
from helpers import get_by_name, get_trainable, permute_together

dataset_name = "lxplus_test2"
dataset = get_by_name(dataset_name, config.WRAPPER_DICT_NAME)['training_data_builder']

training_data, train_pt, event_weight, filtered_data = get_trainable(dataset)
permute_together(training_data, train_pt, event_weight, filtered_data)

xg_reg = xgb.XGBRegressor(objective = 'reg:linear', 
                        learning_rate = .1, 
                        max_depth = 5, 
                        n_estimators = 400,
                        max_bins = 1000,
                        nthread = 30)

train_count = 25000

xg_reg.fit(training_data[:train_count], train_pt[:train_count], sample_weight = event_weight[:train_count])

model_path = os.path.join(config.RESULTS_DIRECTORY, dataset_name, config.MODEL_NAME)
with open(model_path, 'wb') as file:
    pickle.dump(xg_reg, file)

print("------------------------------ Feature Importances ---------------------------")
feature_importances = xg_reg.feature_importances_
for name, importance in zip(np.array(dataset.feature_names)[dataset.trainable_features], feature_importances):
    print(f"{name}:\t {importance}")
print("\n")
print("------------------------------------------------------------------------------")

predicted_pt = np.exp(xg_reg.predict(training_data[train_count:]))

test_dict = {
    "predicted_pt"      : predicted_pt,
    "training_features" : np.array(dataset.feature_names)[dataset.trainable_features],
    "testing_data"      : training_data[train_count:],
    "gen_features"      : np.array(dataset.feature_names)[~dataset.trainable_features],
    "gen_data"          : filtered_data[train_count:][:, ~dataset.trainable_features]
}

prediction_path = os.path.join(config.RESULTS_DIRECTORY, dataset_name, config.PREDICTION_NAME)
with open(prediction_path, 'wb') as file:
    pickle.dump(test_dict, file)