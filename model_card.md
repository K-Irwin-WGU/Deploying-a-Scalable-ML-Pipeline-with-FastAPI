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

####   Column: Datatype 
1. age: integer
2. workclass: string
3. fnlgt: integer 
4. education: string
5. education-num: integer 
6. marital-status: string
7. occupation: string
8. relationship: string
9. race: string
10. sex: string
11. capital-gain: integer 
12. capital-loss: integer 
13. hours-per-week: integer 
14. native-country: string
15. salary: string

## Evaluation Data
The testing data is based on a 20% split of the census.csv file extracted from the US Census database.
The dataset consists of 6513 rows with 15 features.

####   Column: Datatype 
1. age: integer
2. workclass: string
3. fnlgt: integer 
4. education: string
5. education-num: integer 
6. marital-status: string
7. occupation: string
8. relationship: string
9. race: string
10. sex: string
11. capital-gain: integer 
12. capital-loss: integer 
13. hours-per-week: integer 
14. native-country: string
15. salary: string

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