import pandas as pd
from sklearn.preprocessing import StandardScaler


def timefeature(dates):
    dates["hour"] = dates["date"].apply(lambda row: row.hour / 23 - 0.5, 1)  # 一天中的第几小时
    dates["weekday"] = dates["date"].apply(lambda row: row.weekday() / 6 - 0.5, 1)  # 周几
    dates["day"] = dates["date"].apply(lambda row: row.day / 30 - 0.5, 1)  # 一个月的第几天
    dates["month"] = dates["date"].apply(lambda row: row.month / 365 - 0.5, 1)  # 一年的第几天
    return dates[["hour", "weekday", "day", "month"]].values


def get_data(path='./ETT/ETTm2.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    scaler = StandardScaler()
    data = scaler.fit_transform(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values)
    stamp = scaler.fit_transform(timefeature(df))

    train_data = data[:int(0.6 * len(data)), :]
    valid_data = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
    test_data = data[int(0.8 * len(data)):, :]

    train_stamp = stamp[:int(0.6 * len(stamp)), :]
    valid_stamp = stamp[int(0.6 * len(stamp)):int(0.8 * len(stamp)), :]
    test_stamp = stamp[int(0.8 * len(stamp)):, :]

    return [train_data, train_stamp], [valid_data, valid_stamp], [test_data, test_stamp]
