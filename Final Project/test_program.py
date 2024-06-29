'''
Group project Name: US Second-hand Cars Price Prediction Based On Multiple
Features
Members: Maoyi Liao & Ting Wu
This is a test program that help us test the code we wrote previously
especially in the project_code_2.
'''
import project_code_2
import pandas as pd


def test_filtered_train_data(test: pd.DataFrame):
    '''
    Test the filtered_train_data in project_code2, which narrow down
    the features to what we are interested
    and fill in the missing value with None.
    '''
    filtered_data = project_code_2.filtered_train_data(test)
    fill_missing = sum(filtered_data.isna().sum()) == 0
    narrow_column = \
        list(filtered_data.columns) == ['Price', 'Prod. year', 'Mileage',
                                        'Manufacturer', 'Model',
                                        'title_status', 'state',
                                        'Category', 'Fuel type']
    if (fill_missing & narrow_column):
        print("Pass! filtered_train_data is SUCCESSFUL!")
    else:
        print("Oh no! Something wrong with filtered_train_data ~")


def main():
    test = pd.read_csv('test_data.csv')
    test_filtered_train_data(test)


if __name__ == '__main__':
    main()
