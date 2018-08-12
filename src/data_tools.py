import gc
import os
import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.common_paths import get_data_path
from src.general_utilities import batching


def cartesian_pair(df1, df2, **kwargs):
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param df1 dataframe 1
    :param df1 dataframe 2
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of df1 and df2
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res


def cartesian_multiple(df, columns):
    df_cartesian = df.loc[:, [columns[0]]].drop_duplicates()
    for i in range(1, len(columns)):
        df_cartesian = cartesian_pair(df_cartesian, df.loc[:, [columns[i]]].drop_duplicates())

    return df_cartesian


def load_raw_data():
    df_train = pd.read_csv(os.path.join(get_data_path(), "cf", "train.csv"))
    df_test = pd.read_csv(os.path.join(get_data_path(), "cf", "test.csv"))
    df_transactions = pd.read_csv(os.path.join(get_data_path(), "cf", "transactions.csv"))
    df_items = pd.read_csv(os.path.join(get_data_path(), "cf", "items.csv"))
    df_stores = pd.read_csv(os.path.join(get_data_path(), "cf", "stores.csv"))
    df_holiday_events = pd.read_csv(os.path.join(get_data_path(), "cf", "holidays_events.csv"))
    df_oil = pd.read_csv(os.path.join(get_data_path(), "cf", "oil.csv"))
    return df_train, df_test, df_transactions, df_items, df_stores, df_holiday_events, df_oil


def preprocess_data():
    df, df_test, df_transactions, df_items, df_stores, df_holiday_events, df_oil = load_raw_data()
    n_stores = df.store_nbr.nunique()
    n_items = df.item_nbr.nunique()
    n_dates = df.date.nunique()

    le_store_type = LabelEncoder()
    le_item_class = LabelEncoder()
    le_item_family = LabelEncoder()
    le_type_holiday = LabelEncoder()
    le_type_holiday = LabelEncoder()
    le_type_holiday = LabelEncoder()
    le_type_holiday = LabelEncoder()
    le_city = LabelEncoder()
    le_state = LabelEncoder()

    df_oil = df_oil.fillna({"dcoilwtico": 0})

    df_stores = (df_stores
                 .rename(columns={"type": "store_type", "cluster": "store_cluster"})
                 .assign(store_type=lambda x: le_store_type.fit_transform(x.store_type),
                         store_cluster=lambda x: x.store_cluster - 1))

    df_items = (df_items
                .rename(columns={"family": "item_family", "class": "item_class", "perishable": "item_perishable"})
                .assign(item_family=lambda x: le_item_class.fit_transform(x.item_family),
                        item_class=lambda x: le_item_family.fit_transform(x.item_class)))

    df_holiday_events = (df_holiday_events
                         .assign(type=lambda x: le_type_holiday.fit_transform(x.type) + 1)
                         .assign(transferred=lambda x: x.transferred + 0))

    df_national_holidays = (df_holiday_events
                            .loc[df_holiday_events.locale == "National", ["date", "type", "transferred"]]
                            .assign(holiday=1)
                            .rename(columns={"transferred": "national_holiday_transferred",
                                             "holiday": "national_holiday",
                                             "type": "national_holiday_type"}))

    df_regional_holidays = (df_holiday_events
                            .loc[df_holiday_events.locale == "Regional", ["date", "type", "transferred", "locale_name"]]
                            .assign(holiday=1)
                            .rename(columns={"transferred": "regional_holiday_transferred",
                                             "locale_name": "state",
                                             "holiday": "regional_holiday",
                                             "type": "regional_holiday_type"})
                            .drop(["regional_holiday_type", "regional_holiday_transferred"], axis=1))

    df_local_holidays = (df_holiday_events
                         .loc[df_holiday_events.locale == "Local", ["date", "type", "transferred", "locale_name"]]
                         .assign(holiday=1, transferred=lambda x: x.transferred * 2 - 1)
                         .rename(columns={"transferred": "local_holiday_transferred",
                                          "locale_name": "city",
                                          "holiday": "local_holiday",
                                          "type": "local_holiday_type"}))

    df_cartesian = cartesian_multiple(df, ["date", "store_nbr", "item_nbr"])
    df = df_cartesian.merge(df, on=["date", "store_nbr", "item_nbr"], how="left")

    df = (df
          .assign(onpromotion=lambda x: x.onpromotion.astype(float).fillna(-1))
          .fillna({"id": -1, "unit_sales": 0, "onpromotion": -1})
          .merge(df_items, on=["item_nbr"], how="left")
          .merge(df_stores, on=["store_nbr"], how="left")
          .merge(df_national_holidays.drop_duplicates(subset=["date"]), on=["date"], how="left")
          .merge(df_regional_holidays.drop_duplicates(subset=["date", "state"]), on=["date", "state"], how="left")
          .merge(df_local_holidays.drop_duplicates(subset=["date", "city"]), on=["date", "city"], how="left")
          .merge(df_oil, on=["date"], how="left")
          .merge(df_transactions, on=["date", "store_nbr"], how="left")
          .fillna({"national_holiday_type": 0,
                   "national_holiday_transferred": -1,
                   "national_holiday": -1,
                   "regional_holiday_type": 0,
                   "regional_holiday_transferred": -1,
                   "regional_holiday": -1,
                   "local_holiday_type": 0,
                   "local_holiday_transferred": -1,
                   "local_holiday": -1,
                   "transactions": 0,
                   "dcoilwtico": 0})
          .assign(city=lambda x: le_city.fit_transform(x.city),
                  state=lambda x: le_state.fit_transform(x.state))
          .sort_values(by=["store_nbr", "item_nbr", "date"])
          )

    data_cube = df.values.reshape([n_stores * n_items, n_dates, len(df.columns)])

    data_cube = data_cube[data_cube[:, :, 4].sum(axis=1) > 0, :, :]
    gc.collect()
    np.random.seed(655321)
    np.random.shuffle(data_cube)
    gc.collect()

    for i, batch in tqdm(enumerate(batching(data_cube, n=10000, return_incomplete_batches=True))):
        np.save(os.path.join(get_data_path(), "data_cube_{}.npy".format(i)), batch)
    return df, data_cube


def get_batcher(data_cube, batch_size, lag=15, shuffle_present=False, shuffle_periods=100, train=True):
    for batch_cube in batching([data_cube], n=batch_size, return_incomplete_batches=True):
        batch_cube = batch_cube[0]
        if shuffle_present:
            batch_cube = batch_cube[:, :(batch_cube.shape[1] - np.random.randint(0, shuffle_periods))]

        batch = {"store_nbr": batch_cube[:, 0, 1],
                 "item_nbr": batch_cube[:, 0, 2],
                 "unit_sales": batch_cube[:, :-lag, [4]].astype(float),
                 "onpromotion": batch_cube[:, :-lag, [5]],
                 "item_family": batch_cube[:, 0, 6],
                 "item_class": batch_cube[:, 0, 7],
                 "item_perishable": batch_cube[:, 0, 8],
                 "city": batch_cube[:, 0, 9],
                 "state": batch_cube[:, 0, 10],
                 "store_type": batch_cube[:, 0, 11],
                 "store_cluster": batch_cube[:, 0, 12],
                 "national_holiday_type": batch_cube[:, :-lag, [13]],
                 "national_holiday_transferred": batch_cube[:, :-lag, [14]],
                 "national_holiday": batch_cube[:, :-lag, [15]],
                 "regional_holiday": batch_cube[:, :-lag, [16]],
                 "local_holiday_type": batch_cube[:, :-lag, [17]],
                 "local_holiday_transferred": batch_cube[:, :-lag, [18]],
                 "local_holiday": batch_cube[:, :-lag, [19]],
                 "dcoilwtico": batch_cube[:, :-lag, [20]].astype(float),
                 "transactions": batch_cube[:, :-lag, [21]].astype(float),
                 "year": (np.vectorize(lambda x: x[0:4])(batch_cube[:, :-lag, [0]]).astype("float") - 2015) / 2,
                 "month": (np.vectorize(lambda x: x[5:7])(batch_cube[:, :-lag, [0]]).astype("float") - 6.5) / 5.5,
                 "day": (np.vectorize(lambda x: x[8:])(batch_cube[:, :-lag, [0]]).astype("float") - 16) / 15,
                 "dow": (np.vectorize(lambda x: datetime.datetime(int(x[0:4]), int(x[5:7]), int(x[8:])).weekday())(
                     batch_cube[:, :-lag, [0]]).astype("float") - 3) / 3,
                 "year_fut": (np.vectorize(lambda x: x[0:4])(batch_cube[:, -lag:, [0]]).astype("float") - 2015) / 2,
                 "month_fut": (np.vectorize(lambda x: x[5:7])(batch_cube[:, -lag:, [0]]).astype("float") - 6.5) / 5.5,
                 "day_fut": (np.vectorize(lambda x: x[8:])(batch_cube[:, -lag:, [0]]).astype("float") - 16) / 15,
                 "dow_fut": (np.vectorize(lambda x: datetime.datetime(int(x[0:4]), int(x[5:7]), int(x[8:])).weekday())(
                     batch_cube[:, -lag:, [0]]).astype("float") - 3) / 3,
                 "onpromotion_fut": batch_cube[:, -lag:, [5]],
                 "local_holiday_fut": batch_cube[:, -lag:, [19]],
                 "national_holiday_fut": batch_cube[:, -lag:, [15]],
                 "regional_holiday_fut": batch_cube[:, -lag:, [16]]
                 }

        if train:
            batch["target"] = batch_cube[:, -lag:, [4]].astype(float),

        params = {"mean_unit_sales": batch["unit_sales"].mean(axis=1, keepdims=True),
                  "std_unit_sales": batch["unit_sales"].std(axis=1, keepdims=True)}

        params["std_unit_sales"][params["std_unit_sales"] == 0] = 1

        batch["unit_sales"] = (batch["unit_sales"] - params["mean_unit_sales"]) / params["std_unit_sales"]
        batch["target"] = (batch["target"] - params["mean_unit_sales"]) / params["std_unit_sales"]

        yield batch, params
