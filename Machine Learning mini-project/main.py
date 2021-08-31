import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from nn_implementation import NeuralNetworkGridCV, transform_train_val_for_pytorch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import numpy as np
np.random.seed(0)

DATA_FOLDER = "data"
WHR_FILE_PATH = os.path.join(DATA_FOLDER, "2017.csv")
UNDATA_FILE_PATH = os.path.join(DATA_FOLDER, "country_profile_variables.csv")
TARGET_FEATURE = "Happiness.Score"


# Read UNData dataset
und_df = pd.read_csv(UNDATA_FILE_PATH, header=0)
und_df = und_df.rename(columns={"country": "Country"})
und_df = und_df.rename(columns={"Population age distribution (0-14 / 60+ years, %)":
                                "Population age distribution (0-14/60+ years, %)"})
# shape: (229, 50)

# Read WHR dataset
whr_df = pd.read_csv(WHR_FILE_PATH, header=0)
# shape: (155, 12)


# Change names of countries which do not correspond between the two datasets
UND_NAME_TO_WHR_NAME = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Democratic Republic of the Congo": "Congo (Kinshasa)",
    "Congo": "Congo (Brazzaville)",
    "Czechia": "Czech Republic",
    "China, Hong Kong SAR": "Hong Kong S.A.R., China",
    "Iran (Islamic Republic of)": "Iran",
    "The former Yugoslav Republic of Macedonia": "Macedonia",
    "Republic of Moldova": "Moldova",
    "State of Palestine": "Palestinian Territories",
    "Russian Federation": "Russia",
    "Republic of Korea": "South Korea",
    "Syrian Arab Republic": "Syria",
    "United Republic of Tanzania": "Tanzania",
    "United States of America": "United States",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Viet Nam": "Vietnam"
}
for und_name, whr_name in UND_NAME_TO_WHR_NAME.items():
    und_df["Country"].replace(und_name, whr_name, inplace=True)


# Merge two datasets by country
df = pd.merge(
    und_df,
    whr_df,
    on="Country",
    left_index=False,
    right_index=False
)
# shape: (151, 61)


# Drop some wrong or useless columns
COLUMNS_TO_REMOVE = [
    "Country", "Region",                                                    # not useful in training
    "International migrant stock (000/% of total pop.)", "Forested area (% of land area)",
    "Energy supply per capita (Gigajoules)",                                # ambiguous data
    "Net Official Development Assist. received (% of GNI)",                 # all values are -99
    "Happiness.Rank",                                                       # relative to other countries
    "Whisker.high", "Whisker.low", "Health..Life.Expectancy.", "Family",
    "Freedom", "Generosity",                                                # non objective
    "Economy..GDP.per.Capita.",                                             # duplicated
    "Education: Government expenditure (% of GDP)"                          # 14% of instances with NaN
]
for column in COLUMNS_TO_REMOVE:
    df.drop(column, axis=1, inplace=True)
# shape: (151, 46)


# Split columns with two values
TWO_VALUES_COLUMNS = [
    "Labour force participation (female/male pop. %)",
    "Life expectancy at birth (females/males, years)",
    "Population age distribution (0-14/60+ years, %)",
    "Education: Primary gross enrol. ratio (f/m per 100 pop.)",
    "Education: Secondary gross enrol. ratio (f/m per 100 pop.)",
    "Education: Tertiary gross enrol. ratio (f/m per 100 pop.)",
    "Pop. using improved drinking water (urban/rural, %)"
]
for column in TWO_VALUES_COLUMNS:
    for word in column.split():
        if '/' in word:
            value1_name, value2_name = word.strip('(,').split('/')
    column_ = column.replace(value1_name+'/'+value2_name, '{}')
    column1_name = column_.format(value1_name)
    column2_name = column_.format(value2_name)
    columns = pd.DataFrame(df[column].str.split('/').tolist(), columns=[column1_name, column2_name])
    df = pd.concat([df, columns], axis=1)
    df.drop(column, axis=1, inplace=True)
# shape: (151, 53)


# Convert all numeric columns to the appropriate type
for column in df:
    df[column] = pd.to_numeric(df[column], errors='coerce')  # invalid parsing will be set as NaN.


# Transform non-recognized values {'-~0.0', '~0.0', '...'} to NaN:
imputer = KNNImputer(n_neighbors=10, weights="uniform")
assert df[TARGET_FEATURE].isna().sum() == 0
df[:] = imputer.fit_transform(df)


# Aggregate education columns
'''
"Education: Primary gross enrol. ratio (f per 100 pop.)",
"Education: Primary gross enrol. ratio (m per 100 pop.)",
"Education: Secondary gross enrol. ratio (f per 100 pop.)",
"Education: Secondary gross enrol. ratio (m per 100 pop.)",
"Education: Tertiary gross enrol. ratio (f per 100 pop.)",
"Education: Tertiary gross enrol. ratio (m per 100 pop.)"
'''
education_columns = [column for column in df if "Education: " in column]
avg_education_column = df[education_columns].aggregate("mean", axis=1)\
    .to_frame("Education: average gross enrol. ratio (per 100 pop.)")
df = pd.concat([df, avg_education_column], axis=1)
df.drop(education_columns, axis=1, inplace=True)
# shape: (151, 48)


# Discretize target feature
TARGET_CLASSES = ["Very unhappy", "Unhappy", "Ambivalent", "Happy", "Very happy"]
df[TARGET_FEATURE] = pd.cut(df[TARGET_FEATURE], bins=5, labels=range(5))
# ripristina: labels=TARGET_CLASSES, ordered=True) # TODO prova a levare ordered
# plot_ordered_categories_of(TARGET_FEATURE, TARGET_CLASSES)


# X and y
X = df.drop(TARGET_FEATURE, axis=1)
y = df[TARGET_FEATURE]


# Address imbalanced classes
X, y = SMOTE().fit_resample(X, y)


# dataset split into training-validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25)
# training-validation set split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)
# PredefinedSplit for tuning HPs
split_index = [-1 if x in X_train.index else 0 for x in X_train_val.index]
pds = PredefinedSplit(test_fold=split_index)


# HYPER-PARAMETERS TUNING AND TRAINING
print("HP-TUNING")
# Random Forest
print("Random forest HP-tuning")
param_grid_rf = {
    'n_estimators': [100, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 8, 15, 25, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_clf = GridSearchCV(
    RandomForestClassifier(), param_grid_rf, cv=pds, scoring='f1_macro', n_jobs=-1
)
rf_clf.fit(X_train_val, y_train_val)
print(f"Best params: {rf_clf.best_params_}")
print(f"Best score: {rf_clf.best_score_}")
print()

# Decision Tree
print("Decision tree HP-tuning")
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 3, 8, 15, 25, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['auto', 'sqrt', 'log2']
}
dt_clf = GridSearchCV(
    DecisionTreeClassifier(), param_grid_dt, cv=pds, scoring='f1_macro', n_jobs=-1  # TODO prova con f1-score
)
dt_clf.fit(X_train_val, y_train_val)
print(f"Best params: {dt_clf.best_params_}")
print(f"Best score: {dt_clf.best_score_}")
print()


# Neural network
print("Neural network HP-tuning")
X_train_scaled = preprocessing.scale(X_train)
X_val_scaled = preprocessing.scale(X_val)       # scaling data before fitting the nn
param_grid_nn = {
    'hidden_layer_sizes': [(20,), (25, 10), (100,), (100, 500, 50), (100, 50, 10), (10, 100, 10)],
    'activations': ['relu', 'tanh'],
    'early_stop_patiences': [1, 5, 10, 50]
}
nn_classifier = NeuralNetworkGridCV(
    n_features=X.shape[1], param_grid=param_grid_nn
)
train_val_for_pytorch = transform_train_val_for_pytorch(X_train_scaled, y_train, X_val_scaled, y_val)
nn_classifier.fit(*train_val_for_pytorch)
print(f"Best params: {nn_classifier.best_params_}")
print(f"Best score: {nn_classifier.best_score_}")


# Evaluation
print("\nEVALUATION Random Forest")
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))