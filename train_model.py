import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the cencus.csv data
project_path = os.path.dirname(__file__)
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# Split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Save Data to CSV
train_path = data_path = os.path.join(project_path, "data", "train_data.csv")
test_path = data_path = os.path.join(project_path, "data", "test_data.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Use the process_data function provided to process the data.
# Use the train dataset 
# Use training=True
# No need to pass encoder and lb as input
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# Save the model and the encoder
project_subfolder = "model"
model_file = "model.pkl"
encoder_file = "encoder.pkl"

model_path = os.path.join(project_path, project_subfolder, model_file)
save_model(model, model_path)
print(f'Model saved to {project_subfolder}/{model_file}')

encoder_path = os.path.join(project_path, project_subfolder, encoder_file)
save_model(encoder, encoder_path)
print(f'Model saved to {project_subfolder}/{encoder_file}')


# Load the model
model = load_model(
    model_path
) 

# Use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    # use test, col and slicevalue as part of the input
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
                
        p, r, fb = performance_on_categorical_slice(
            data = test, 
            column_name = col, 
            slice_value = slicevalue,
            categorical_features = cat_features, 
            label = "salary",
            encoder = encoder, 
            lb = lb, 
            model = model
        )
        
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
        


