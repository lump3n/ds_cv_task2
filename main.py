from graphics import plot_distribution, plot_linear_regr_line_with_distr, plot_poly_regr_line_with_distr
from regression import LinearRegression, PolynomialRegression, SVMRegression
from preprocessing import df_typing, split_df_to_train_test_sets, filter_feature, \
    time_conversion, sampling_df
import pandas as pd
import numpy as np


def build_linear_regression(df):
    x_train, x_test, y_train, y_test = split_df_to_train_test_sets(df)
    plot_distribution(x_test, y_test, 'Нагрузка системы в течении суток')
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    y_pred = linear_model.predict(x_test)
    mse, r2 = linear_model.calculate_metrics(y_test, y_pred)
    print("Метрики линейной регрессии:")
    linear_model.print_metrics(mse, r2)
    plot_linear_regr_line_with_distr(x_test, y_test, y_pred, 'График линейной регрессии')


def build_polynomial_regression(df):
    x_train, x_test, y_train, y_test = split_df_to_train_test_sets(df)
    plot_distribution(x_test, y_test, 'Нагрузка системы в течении суток после фильтрации данных')
    poly_model = PolynomialRegression()
    sorted_x_test = np.sort(x_test, axis=0)
    poly_x_train, poly_x_test = poly_model.polynomialize_features(x_train, sorted_x_test, degree=3)
    poly_model.fit(poly_x_train, y_train)
    y_pred = poly_model.predict(poly_x_test)
    mse, r2 = poly_model.calculate_metrics(y_test, y_pred)
    print('\n', "Метрики полиномиальной регрессии:")
    poly_model.print_metrics(mse, r2)
    plot_poly_regr_line_with_distr(x_test, y_test, sorted_x_test, y_pred, 'График полиномиальной регрессии')


def build_svm_regression(df):
    x_train, x_test, y_train, y_test = split_df_to_train_test_sets(df)
    svr_model = SVMRegression()
    svr_model.fit(x_train, y_train.ravel())
    sorted_x_test = np.sort(x_test, axis=0)
    y_pred = svr_model.predict(sorted_x_test)
    mse, r2 = svr_model.calculate_metrics(y_test, y_pred)
    print('\n', "Метрики SVM регрессии:")
    svr_model.print_metrics(mse, r2)
    plot_poly_regr_line_with_distr(x_test, y_test, sorted_x_test, y_pred, 'График SVM регрессии')


def main():
    # Считывание данных из txt файла и предобработка
    dataframe = pd.read_csv('time_messagees.txt', header=None, names=['time', 'num_of_messages'])
    df = df_typing(dataframe)
    df = df.set_index('time')
    df = sampling_df(df, '5s')
    df = time_conversion(df)

    # Построение линейной регрессии
    build_linear_regression(df)
    # Построение полиномиальной регрессии
    df = filter_feature(df)
    build_polynomial_regression(df)
    # Построение SVM регрессии
    build_svm_regression(df)


if __name__ == '__main__':
    main()
