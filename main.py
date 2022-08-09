import json
import math
from random import random

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def read_from_file(file_path: str, encoding: str) -> object:
    data_framed = pd.read_csv(file_path, encoding=encoding)
    return data_framed


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as config_file:
        return json.loads(config_file.read())


# 1.2
def get_total_missing_percentage(data_frame: pd.DataFrame) -> float:
    null_count = data_frame.isnull().sum().sum()
    total_count = null_count + data_frame.count().sum()
    return (null_count / total_count) * 100


# 1.3 + 1.4
def validate_file(data_frame: pd.DataFrame) -> bool:
    miss_percentage = get_total_missing_percentage(data_frame)
    # TODO :: remove comment before real run
    # if miss_percentage < 10:
    #     return False
    # if miss_percentage == 0 or contains_valid_formats(data_frame):
    #     return False
    return True


# 1.4
def contains_valid_formats(data_frame: pd.DataFrame) -> bool:
    for column in data_frame.select_dtypes(include=np.number).columns.tolist():
        invalid_rows = get_numeric_column_invalid_rows(column, data_frame)
        if len(invalid_rows) > 0:
            return False
    for column in data_frame.select_dtypes(include=np.object).columns.tolist():
        invalid_rows = get_str_column_invalid_rows(column, data_frame)
        if len(invalid_rows) > 0:
            return False
    return True


def get_numeric_column_invalid_rows(column_name, data_frame):
    invalid_row_indexes = []
    for index in data_frame.index:
        if (not pd.isnull(data_frame.at[index, column_name])) and (
                type(data_frame.at[index, column_name]) == str or type(
            data_frame.at[index, column_name]) == bool):
            invalid_row_indexes.append(index)
    return invalid_row_indexes


def get_str_column_invalid_rows(column_name, data_frame):
    invalid_row_indexes = []
    for index in data_frame.index:
        # invalid type (type float for unknown reason)
        if (not pd.isnull(data_frame.at[index, column_name])) and (
                type(data_frame.at[index, column_name]) == int or type(
            data_frame.at[index, column_name]) == float or type(
            data_frame.at[index, column_name]) == bool):
            invalid_row_indexes.append(index)
    return invalid_row_indexes


# 1.5
def print_data_from_df(data_frame: pd.DataFrame) -> None:
    print("first 5 lines in df: ")
    print(data_frame.head(5))
    print("last 5 lines in df: ")
    print(data_frame.tail(5))
    print("middle 5 lines in df: ")
    print(data_frame.iloc[100:105])


# 1.6
def print_df_description(data_frame: pd.DataFrame) -> None:
    print("description of data frame:")
    print(data_frame.describe(include='all'))  # presentation p7


# 1.8
def merge_outer_join_df(data_frame_1: pd.DataFrame, data_frame_2: pd.DataFrame) -> pd.DataFrame:
    common_column = 'tmp'
    while common_column in data_frame_1.columns or common_column in data_frame_2.columns:  # if tmp in one of the data
        # frames, then we pick a different name to the added column
        common_column = str(random.randint(0, 9000))

    data_frame_1[common_column] = 1
    data_frame_2[common_column] = 1

    merged_df = data_frame_1.merge(data_frame_2, on=common_column, how='outer')
    merged_df = merged_df.drop(columns=[common_column])
    data_frame_2.drop(columns=[common_column])
    data_frame_1.drop(columns=[common_column])
    validate_and_print_df(merged_df)  # validate as we did for both df separately
    return merged_df


# 1.9
def split_df(merged_data_frame: pd.DataFrame, df1_columns: list, df2_columns: list) -> object:
    merged_columns = list(merged_data_frame.columns.values)

    df1_columns = [col for col in df1_columns if col in merged_columns]
    df2_columns = [col for col in df2_columns if col in merged_columns]

    df1 = merged_data_frame[df1_columns]  # selects out of merged df only columns received for df1
    df2 = merged_data_frame[df2_columns]
    return df1, df2


def validate_and_print_df(data_frame: pd.DataFrame):
    df_valid = validate_file(data_frame)
    print('q 1.3 + 1.4')
    if not df_valid:
        print('invalid file')
        exit(1)
    print('q 1.5')
    print_data_from_df(data_frame)
    print('q 1.6')
    print_df_description(data_frame)


# 2.4
def normalize_column(data_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    max_value = data_frame[column].max()
    for i in data_frame.index:
        data_frame.at[i, column] = math.fabs(data_frame.at[i, column]) / max_value
    return data_frame


# 2.1
def clean_column_invalid_numeric_type_data(data_frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    invalid_row_indexes = get_numeric_column_invalid_rows(column_name, data_frame)
    for index in list(reversed(invalid_row_indexes)):
        data_frame = data_frame.drop(data_frame.index[index])
    return data_frame


# 2.2
def clean_column_invalid_str_type_data(data_frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data_frame[column_name] = data_frame[column_name].replace([True, False], 1)
    invalid_row_indexes = get_str_column_invalid_rows(column_name, data_frame)
    for index in list(reversed(invalid_row_indexes)):
        data_frame = data_frame.drop(data_frame.index[index])
    return data_frame


# 2.3
def update_number_null_values(data_frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    mean_value = data_frame[column_name].mean()
    data_frame[column_name] = data_frame[column_name].fillna(value=mean_value)
    return data_frame


def update_str_null_values(data_frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    frequent_value = data_frame[column_name].value_counts().idxmax()
    data_frame[column_name] = data_frame[column_name].fillna(value=frequent_value)
    return data_frame


# 2.5
def handle_duplicates(data_frame: pd.DataFrame) -> None:
    if data_frame.duplicated().count() == 0:
        for i in range(10):
            data_frame = data_frame.append(data_frame.at(0))
    print_duplicates_and_remove(data_frame)


def print_duplicates_and_remove(data_frame: pd.DataFrame) -> None:
    print("duplicates values: ")
    print(data_frame.duplicated())
    data_frame.drop_duplicates()


# 4.1
def k_mean_on_df(data_array: object) -> None:
    k: int = get_k_by_elbow_prediction(data_array)
    present_k_mean_graph(data_array, k)


def present_k_mean_graph(data_array, k: int, title: str = "k mean graph", x_label: str = "x", y_label: str = "y"):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(data_array)
    plt.plot()
    plt.scatter(data_array[y_kmeans == 0, 0], data_array[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(data_array[y_kmeans == 1, 0], data_array[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(data_array[y_kmeans == 2, 0], data_array[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(data_array[y_kmeans == 3, 0], data_array[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(data_array[y_kmeans == 4, 0], data_array[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(data_array[y_kmeans == 5, 0], data_array[y_kmeans == 5, 1], s=100, c='yellow', label='Cluster 6')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', label='Centroids')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def get_k_by_elbow_prediction(data_array) -> int:
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data_array)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('wcss')
    plt.show(block=False)
    plt.pause(0.1)
    print("based on graph -> choose the desired k based on elbow method")
    chosen_k = int(input())
    return chosen_k


# 4.2
def linear_regression(x: list, y: list) -> None:
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1 / len(x) +
                              (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    fig, ax = plt.subplots()
    ax.plot(x, y_est, '-')
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, 'o', color='tab:green')
    fig.show()
    plt.waitforbuttonpress()


def run_q_4(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    # k_mean_on_df(df1.iloc[:, [4, 5]].values)
    # k_mean_on_df(df2.iloc[:, [6, 7]].values)
    linear_regression(df1['weekly_cases'], df1['biweekly_cases'])
    linear_regression(df2['deaths_per_100000'], df2['confirmed_per_100000'])


def run_q_3(df1: pd.DataFrame, df2: pd.DataFrame):
    # adding a year-month column to have the option to work on the data with more perspectives
    df1['YearMonth'] = pd.to_datetime(df1['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
    df2['YearMonth'] = pd.to_datetime(df2['last_update']).apply(
        lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
    sns.set_style('dark')
    # 1 deaths by Month - general
    sns.lineplot(x='YearMonth', y='total_deaths', data=df1, palette='winter_r')

    # 2 deaths by month - general us
    sns.lineplot(x='YearMonth', y='deaths', data=df2, palette='winter_r')

    # 3 pie - percentage by country for cases
    plt.figure()
    plt.pie(df1.groupby('location').sum()['total_cases'],
            labels=list(df1.groupby('location').sum()['total_cases'].keys()), autopct='%1.1f%%')
    # 4 pie - percentage by country for deaths
    plt.figure()
    plt.pie(df1.groupby('location').sum()['total_deaths'],
            labels=list(df1.groupby('location').sum()['total_deaths'].keys()), autopct='%1.1f%%')

    # 5 pie - percentage by state in the us for cases
    plt.pie(df2.groupby('state').sum()['confirmed'], labels=list(df2.groupby('state').sum()['confirmed'].keys()),
            autopct='%1.1f%%')
    # 6 pie - percentage by state in the us for deaths
    plt.pie(df2.groupby('state').sum()['deaths'], labels=list(df2.groupby('state').sum()['deaths'].keys()),
            autopct='%1.1f%%')

    # 6.5 bars - cases per month in general
    df1.groupby(['YearMonth'])['total_cases'].sum().plot(kind='bar')

    # 7 bars - cases + deaths per month in general
    df1.groupby(['YearMonth'])['total_cases', 'total_deaths'].sum().plot(kind='bar')

    # 8 bars - cases compared to population by state
    df2.groupby('state')['total_population', 'confirmed'].sum().plot(kind='bar')

    # 9 cases by dates in the us - general
    df2.plot(kind='scatter', x='last_update', y='confirmed')

    # 10 deaths by state and location type
    fig1 = px.scatter(df2, x="deaths_per_100000", y="state", color="location_type")
    fig1.show()

    # 11
    sns.set_style("whitegrid")
    sns.pairplot(df1, hue="location", size=3)

    # 12 - how many rows we get per each state for this experiment
    sns.countplot(x='state', data=df2)

    # 13 - how many rows we get per each country for this experiment
    sns.countplot(x='location', data=df1)


def run_q_2(df1: pd.DataFrame, df2: pd.DataFrame):
    # cleaning column invalid types rows
    for column in df1.select_dtypes(include=np.number).columns.tolist():
        clean_column_invalid_numeric_type_data(df1, column)
    for column in df2.select_dtypes(include=np.number).columns.tolist():
        clean_column_invalid_numeric_type_data(df2, column)
    for column in df1.select_dtypes(include=np.object).columns.tolist():  # object refers to str -> no type str in df
        clean_column_invalid_str_type_data(df1, column)
    for column in df2.select_dtypes(include=np.object).columns.tolist():
        clean_column_invalid_str_type_data(df2, column)

    # make sure no nulls (nan) in df
    for column in df1.select_dtypes(include=np.number).columns.tolist():
        update_number_null_values(df1, column)
    for column in df2.select_dtypes(include=np.number).columns.tolist():
        update_number_null_values(df2, column)
    for column in df1.select_dtypes(include=np.object).columns.tolist():
        update_str_null_values(df1, column)
    for column in df2.select_dtypes(include=np.object).columns.tolist():
        update_str_null_values(df2, column)

    # Normalizing columns based on columns received in config file
    columns_to_normalize_df1 = config['data_frames'][0]['columns_to_normalize']
    columns_to_normalize_df2 = config['data_frames'][1]['columns_to_normalize']
    # before trying to take data
    for column_to_normalize_df1 in columns_to_normalize_df1:
        normalize_column(df1, column_to_normalize_df1)
    for column_to_normalize_df2 in columns_to_normalize_df2:
        normalize_column(df2, column_to_normalize_df2)

    # remove all duplicate lines
    handle_duplicates(data_frame=df1)
    handle_duplicates(data_frame=df2)


def run_q_1():
    df1_file_path = config['data_frames'][0]['path']
    df2_file_path = config['data_frames'][1]['path']
    data_frame_1: pd.DataFrame = read_from_file(df1_file_path, 'utf-8')
    data_frame_2: pd.DataFrame = read_from_file(df2_file_path, 'utf-8')
    validate_and_print_df(data_frame_1)
    validate_and_print_df(data_frame_2)
    # TODO :for our usage - remove below line before submission
    return data_frame_1, data_frame_2
    # TODO :for our usage - restore below lines before submission
    # merged_data_frame = merge_outer_join_df(data_frame_1, data_frame_2)
    # df1, df2 = split_df(merged_data_frame, data_frame_1.columns, data_frame_2.columns)
    # return df1, df2


#
# df = pd.DataFrame.from_dict({'Name': ['May21', True, False, -5, 'Hello', 'Girl90', 90],
#                              'Volume': [23, 12, 11, 34, 56, 1, 1],
#                              'Value': [21321, 12311, 4435, 3, 2, 454, 654654]})
# df1 = pd.DataFrame(df)
# to_be_dropped = []
# df['Name'] = df['Name'].replace([True, False], 1)
# # df['Name'] = df['Name'].drop([item for item in df['Name'] if type(item) == int])
# for i in df.index:
#     if is_numeric_dtype(df.at[i, 'Name']):
#         to_be_dropped.append(i)
# for i in list(reversed(to_be_dropped)):
#     df = df.drop(df.index[i])
#
#
# print(df)

config = load_config('data_science_config.json')
df1, df2 = run_q_1()
run_q_2(df1, df2)
run_q_3(df1, df2)
run_q_4(df1, df2)
input()
