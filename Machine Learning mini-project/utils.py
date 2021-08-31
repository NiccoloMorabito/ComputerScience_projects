import matplotlib.pyplot as plt
import pandas as pd

REGION_TO_CONTINENT = {
    'Caribbean': 'North-America',
    'CentralAmerica': 'South-America',
    'CentralAsia': 'Asia',
    'EasternAfrica': 'Africa',
    'EasternAsia': 'Asia',
    'EasternEurope': 'Europe',
    'MiddleAfrica': 'Africa',
    'NorthernAfrica': 'Africa',
    'NorthernAmerica': 'North-America',
    'NorthernEurope': 'Europe',
    'Oceania': 'Oceania',
    'South-easternAsia': 'Asia',
    'SouthAmerica': 'South-America',
    'SouthernAfrica': 'Africa',
    'SouthernAsia': 'Asia',
    'SouthernEurope': 'Europe',
    'WesternAfrica': 'Africa',
    'WesternAsia': 'Asia',
    'WesternEurope': 'Europe'
}


def get_unknown_symbols(df):
    unknown_symbols = set()
    for column in df:
        if df[column].dtype == object and column != "country" and column != "Region" and column != "Country":
            for el in df[column]:
                try:
                    pd.to_numeric(el)
                except:
                    if "/" not in el:
                        unknown_symbols.add(el)
    return unknown_symbols


def encode_region(df: pd.DataFrame, to_continent: bool):
    if to_continent:
        for region, continent in REGION_TO_CONTINENT.items():
            df["Region"].replace(region, continent, inplace=True)

    # obtaining dummy columns to encode 'Region' feature
    dummy_regions = df["Region"].str.get_dummies(sep=',').add_prefix('Is in ')
    df = pd.concat([df, dummy_regions], axis=1)
    df.drop(["Region"], axis=1, inplace=True)
    return df


def train_and_plot(classifier, X_train, y_train, X_val, y_val):
    title = "Test vs validation f1-macro scores"
    logs = classifier.fit(X_train, y_train, X_val, y_val)
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(logs["train_history"]))), logs["train_history"], label="Train f1-macro score")
    plt.plot(list(range(len(logs["valid_history"]))), logs["valid_history"], label="Validation f1-macro score")

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("f1-macro score")
    plt.legend(loc="upper right")

    plt.show()


def plot_ordered_categories_of(df, feature, ordered_list):
    ordered_feature = pd.Categorical(df[feature], categories=ordered_list)
    ordered_feature.value_counts().plot(kind='bar')
    plt.show()


def print_dtypes(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.dtypes)
