'''
Group project Name: US Second-hand Cars Price Prediction Based On Multiple
Features
Members: Maoyi Liao & Ting Wu
This is the second python file in our project. In this project_code_2.py, we
did the deeper analysis of our data, including drawing plots for data analysis
and the machine learning.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.stats import skew
from sklearn.metrics import mean_squared_error

sns.set()


def choose_brand(data: pd.DataFrame) -> list:
    '''
    Returns a list of brand that have the largest mean mile.
    '''
    mean_mile = data.groupby('Manufacturer')['Mileage'].mean()
    not_whole = (mean_mile - mean_mile.astype(int)) != 0
    mean_mile = mean_mile[not_whole]
    top_9 = mean_mile.nlargest(9)
    return top_9.index.tolist()


def plot_price_mileage(data: pd.DataFrame, brand: list):
    '''
    Plots a graph that show the relationship between car price and miles.
    '''
    data = data[data['Manufacturer'].isin(brand)]
    plot = sns.relplot(x='Mileage', y='Price', data=data, hue='Prod. year',
                       col='Manufacturer', col_wrap=3)
    plot.set_axis_labels('Mileage', 'Price')
    plot.fig.suptitle('Relationship between mileage and price of used cars',
                      fontsize='x-large', fontweight='bold')
    plot.fig.subplots_adjust(top=0.94)
    plt.savefig('price_vs_mileage.png', bbox_inches='tight')
    plt.show()


def train(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns a dataframe (used to train model later).
    '''
    train = data.iloc[range(6000), :]
    return train


def test(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns a dataframe used to train model and
    test our choice on models later.
    '''
    test = data.iloc[6001:, :]
    return test


def numerical_corr(train: pd.DataFrame):
    '''
    Plots bargraphs that show the price distribution before
    and after normalization. Prints out the correlation between
    numerical features and price.
    '''
    train_num = train.select_dtypes(exclude=object)
    price = train['Price']
    sns.histplot(data=price)
    plt.savefig("price_histogram.png")
    plt.show()

    train_num["log_price"] = np.log(train["Price"])
    log_price = train_num['log_price']
    sns.histplot(log_price)
    plt.savefig("log_price_histogram.png")
    plt.show()

    fig, axs = plt.subplots(nrows=2)
    sns.histplot(data=price, ax=axs[0])
    sns.histplot(log_price, ax=axs[1])
    plt.savefig("both_price_histogram.png")
    plt.show()

    corr_price = train_num.corr()["Price"]
    corr_price = corr_price.sort_values(ascending=False)
    print(corr_price)


def categorical_corr(train: pd.DataFrame):
    '''
    Plots boxplots that show the relation between categorical features and
    price.
    '''
    train_cat = train.select_dtypes(include=object)
    train_cat = train_cat.fillna("None")  # fill missing value with None

    # List the number of each categorical variable
    cols = list(train_cat.columns)
    for col in cols:
        print(col)
        print(train_cat[col].value_counts())
        print("="*50)

    # Create boxplots to explore relation between
    # log_price and categorical features
    train_cat["log_price"] = np.log(train["Price"])
    sns.boxplot(data=train_cat, x="Manufacturer", y="log_price")
    plt.xticks(rotation=-90)
    plt.savefig('boxplot_Manufacturer.png', bbox_inches='tight')
    plt.show()

    # print(train_cat.info()) 13 features, subplot = 12
    n_rows = 4
    n_cols = 3
    cols = train_cat.columns

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 30))
    for r in range(n_rows):
        for c in range(n_cols):
            i = r*n_cols + c
            if i < 13:
                sns.boxplot(data=train_cat, x=cols[i], y="log_price",
                            ax=axs[r][c])
                axs[r][c].set_xticklabels(axs[r][c].get_xticklabels(),
                                          rotation=-90, ha='right')
    plt.savefig('boxplots_cat_feature.png', bbox_inches='tight')
    plt.show()


def filtered_train_data_log(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Returns the dataframe containing the most important columns chosen after
    previous analyzing.
    '''
    data["log_price"] = np.log(data["Price"])
    filtered_data_log = data[["log_price", "Prod. year", "Mileage",
                              "Manufacturer", "Model", "title_status",
                              "state", "Category", "Fuel type"]]
    filtered_data_log = filtered_data_log.fillna("None")
    return filtered_data_log


def machine_learning(train: pd.DataFrame):
    '''
    Creates three models for machine learning the data and compares
    their behavior.
    '''
    numeric_feats = list(train.dtypes[train.dtypes != "object"].index)
    skewness = train[numeric_feats].apply(lambda x: skew(x))
    print("skewness", skewness)

    skewed_feats = skewness.index
    print("df before log", train)
    for i in range(len(skewed_feats)):
        train[skewed_feats[i]] = np.log1p(train[skewed_feats[i]])
    print("df after log", train)

    train = pd.get_dummies(train)
    print("df after get dummies", train.info)

    # Process the Data
    features = train.drop(columns=['log_price'])
    labels = train["log_price"]
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=3)

    # =====Simple Linear Model=====
    model_lm = LinearRegression()
    model_lm.fit(features_train, labels_train)
    # find parameters and the most impactful variable, sort with argsort
    maxcoef = np.argsort(-np.abs(model_lm.coef_))
    coef = model_lm.coef_[maxcoef]
    for i in range(0, 10):
        print("{:.<025} {:< 010.4e}".format(train.columns[maxcoef[i]],
              coef[i]))

    lm_predicted = model_lm.predict(features_test)
    # Mean squared error
    error_lm = mean_squared_error(labels_test, lm_predicted)
    print("Linear regression Mean squared error", error_lm)

    # =====Lasso Regression model=====
    model_lasso = Lasso(alpha=0.1)
    model_lasso.fit(features_train, labels_train)

    lasso_predicted = model_lasso.predict(features_test)
    # Mean squared error
    error_lasso = mean_squared_error(labels_test, lasso_predicted)
    print("Lasso Regression Mean squared error", error_lasso)

    # =====Ridge Regression model=====
    model_ridge = Ridge(alpha=0.1)
    model_ridge.fit(features_train, labels_train)

    ridge_predicted = model_ridge.predict(features_test)
    # Mean squared error
    error_ridge = mean_squared_error(labels_test, ridge_predicted)
    print("Ridge Regression Mean squared error", error_ridge)


def test_smaller(test: pd.DataFrame):
    '''
    Creates models to machine learning again with new data.
    '''
    machine_learning(test)


def main():
    combined_data = pd.read_csv('combined_data.csv')
    brand = choose_brand(combined_data)
    plot_price_mileage(combined_data, brand)
    train_data = train(combined_data)
    print(train_data.columns)
    test_data = test(combined_data)
    test_data.to_csv('test_data.csv')
    numerical_corr(train_data)
    categorical_corr(train_data)
    filtered_data_log = filtered_train_data_log(train_data)
    machine_learning(filtered_data_log)
    test_data_log = filtered_train_data_log(test_data)
    test_smaller(test_data_log)


if __name__ == '__main__':
    main()
