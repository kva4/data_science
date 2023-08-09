# ------------------  HTTP: для парсингу сайтів ----------------------------
'''

Приклад
парсингу сайтів із збереженням інформації до файлів різного формату
df.to_csv("output.csv")
df.to_excel("output.xlsx")
df.to_json("output.json")

'''

import requests
import pandas as pd
import json
import datetime
from datetime import datetime as dt
from pathlib import Path

# the url has timespan that valids for 15 minutes. After that you need to get new url
URL_TEMPLATE = "https://my.meteoblue.com/dataset/query?json=%7B%22units%22%3A%7B%22temperature%22%3A%22CELSIUS%22%2C%22velocity%22%3A%22KILOMETER_PER_HOUR%22%2C%22length%22%3A%22metric%22%2C%22energy%22%3A%22watts%22%7D%2C%22geometry%22%3A%7B%22type%22%3A%22MultiPoint%22%2C%22coordinates%22%3A%5B%5B7.57327%2C47.5584%2C279%5D%5D%2C%22locationNames%22%3A%5B%22%5Cu0411%5Cu0430%5Cu0437%5Cu0435%5Cu043b%5Cu044c%22%5D%7D%2C%22format%22%3A%22highcharts%22%2C%22timeIntervals%22%3A%5B%222022-01-01T%2B02%3A00%5C%2F2023-08-01T%2B02%3A00%22%5D%2C%22timeIntervalsAlignment%22%3A%22none%22%2C%22queries%22%3A%5B%7B%22domain%22%3A%22ERA5T%22%2C%22timeResolution%22%3A%22daily%22%2C%22codes%22%3A%5B%7B%22code%22%3A11%2C%22level%22%3A%222+m+elevation+corrected%22%2C%22aggregation%22%3A%22max%22%7D%2C%7B%22code%22%3A11%2C%22level%22%3A%222+m+elevation+corrected%22%2C%22aggregation%22%3A%22min%22%7D%2C%7B%22code%22%3A11%2C%22level%22%3A%222+m+elevation+corrected%22%2C%22aggregation%22%3A%22mean%22%7D%5D%7D%5D%7D&apikey=5838a18e295d&ts=1690880542&sig=41107d68c738ef7d835e850b76606d2f"

def get_data():
    path = Path('./libs/basel_output.csv')
    df = None
    if path.exists():
        df = pd.read_csv(path)
        print(df)
    else:
        df = pd.DataFrame(data=__get_meteo_data(URL_TEMPLATE))
        df.to_csv(path, index=False)
        print(df)
    return df

def get_clean_data():
    meteo_data = get_data()
    #check for null values
    print("null values")
    print(meteo_data.isnull().sum())
    #drop null values
    meteo_data = meteo_data.dropna().reset_index(drop=True)

    return meteo_data


def __get_meteo_data (URL_TEMPLATE):
    '''
    load meteo data of Basel as json from https://www.meteoblue.com/ru/%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D0%B0/archive/export

    :param URL_TEMPLATE: URL Site
    :return: class 'dict'
    '''

    r = requests.get(URL_TEMPLATE)
    result_list = {'date': [], 'max': [], 'min': [], 'mean': []}
    print(r.status_code)
    print(r.text)
    response = json.loads(r.text)
    result_list['max'] = response["series"][0]["data"]
    result_list['min'] = response["series"][1]["data"]
    result_list['mean'] = response["series"][2]["data"]

    date=dt.fromtimestamp(response["series"][0]["pointStart"]/1000)
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    while date <= today:
        result_list['date'].append(date)
        date += datetime.timedelta(milliseconds=response["series"][0]["pointInterval"])

    print(type(result_list))

    return result_list


