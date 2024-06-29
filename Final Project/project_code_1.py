'''
Group project Name: US Second-hand Cars Price Prediction Based On Multiple
Features
Members: Maoyi Liao & Ting Wu
This is the first python file in our project. In this project_code_1.py, we
combined the two datasets and did the basic and general cleaning for our data.
'''
import pandas as pd


def clean_combine_data(data_1: pd.DataFrame,
                       data_2: pd.DataFrame) -> pd.DataFrame:
    '''
    Saved data into a new csv file after filtering.
    '''
    data_1 = data_1.loc[:, ['price', 'brand', 'model', 'year', 'title_status',
                            'mileage', 'color', 'state']]
    rename_columns = {'price': 'Price', 'brand': 'Manufacturer',
                      'model': 'Model', 'year': 'Prod. year',
                      'mileage': 'Mileage', 'color': 'Color'}
    data_1.rename(columns=rename_columns, inplace=True)
    data_2 = data_2.drop(columns='ID')
    data_2 = data_2.dropna()
    over_1000 = data_1['Price'] >= 1000
    under_100000 = data_1['Price'] <= 100000
    over_10 = data_1['Mileage'] >= 10
    upperbound = data_1['Mileage'] <= 300000
    data_1 = data_1[over_1000 & under_100000 & upperbound & over_10]
    data_2['Mileage'] = data_2['Mileage'].str.split(' ').str[0]
    data_2['Mileage'] = data_2['Mileage'].astype(int)
    data_2['Engine volume'] = data_2['Engine volume'].str.split(' ').str[0]
    data_2['Engine volume'] = data_2['Engine volume'].astype(float)
    over_1000 = data_2['Price'] >= 1000
    under_100000 = data_2['Price'] <= 100000
    over_10 = data_2['Mileage'] > 10
    upperbound = data_2['Mileage'] <= 300000
    data_2 = data_2[over_1000 & under_100000 & upperbound & over_10]
    data_2['Manufacturer'] = data_2['Manufacturer'].str.lower()
    combined_data = pd.concat([data_1, data_2])
    combined_data.to_csv('combined_data.csv')


def main():
    DATA_1 = 'USA_cars_datasets.csv'
    DATA_2 = 'car_price_prediction.csv'
    data_1 = pd.read_csv(DATA_1)
    data_2 = pd.read_csv(DATA_2, na_values=['-'])
    clean_combine_data(data_1, data_2)


if __name__ == '__main__':
    main()
