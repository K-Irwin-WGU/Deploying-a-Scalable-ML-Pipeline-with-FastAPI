# Model Card

For additional information, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model created by Keith Irwin
It is Random Forest Classifier using the default hyperparameters in scikit-learn 1.5.1.

## Intended Use
This model should be used to predict the probability of a person earning over $50k based on attributes in the US census data. 
The model is intended for academics or research purposes.

## Training Data
The training data is based on an 80% split of the census.csv file extracted from the US Census database.
The dataset consists of 26048 rows with 15 features.

 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             26048 non-null  int64 
 1   workclass       26048 non-null  object
 2   fnlgt           26048 non-null  int64 
 3   education       26048 non-null  object
 4   education-num   26048 non-null  int64 
 5   marital-status  26048 non-null  object
 6   occupation      26048 non-null  object
 7   relationship    26048 non-null  object
 8   race            26048 non-null  object
 9   sex             26048 non-null  object
 10  capital-gain    26048 non-null  int64 
 11  capital-loss    26048 non-null  int64 
 12  hours-per-week  26048 non-null  int64 
 13  native-country  26048 non-null  object
 14  salary          26048 non-null  object

## Evaluation Data
The testing data is based on a 20% split of the census.csv file extracted from the US Census database.
The dataset consists of 6513 rows with 15 features.

 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             6513 non-null   int64 
 1   workclass       6513 non-null   object
 2   fnlgt           6513 non-null   int64 
 3   education       6513 non-null   object
 4   education-num   6513 non-null   int64 
 5   marital-status  6513 non-null   object
 6   occupation      6513 non-null   object
 7   relationship    6513 non-null   object
 8   race            6513 non-null   object
 9   sex             6513 non-null   object
 10  capital-gain    6513 non-null   int64 
 11  capital-loss    6513 non-null   int64 
 12  hours-per-week  6513 non-null   int64 
 13  native-country  6513 non-null   object
 14  salary          6513 non-null   object

## Metrics
The model was evaluated using three metrics: Precision, Recall, and F1.
- Precision: 0.7422
- Recall: 0.6088
- F1: 0.6689

## Ethical Considerations
The model is trained on a random split of the census data.
The model is not biased towards any particular segment of the data.

## Caveats and Recommendations
The model reflects and acurate patrial based on the data provided at the given time.
It is reccomended that users provide a fresh dataset for more up-to-date results.