import os
from feature_generation_temp import preprocessing


def predict(x_train, y_train, x_test):
    pass


if __name__ == '__main__':
    file_path = os.getcwd() + '/data_files/'

    X_train, y_train, X_test = preprocessing(file_path)
    predict(X_train, y_train, X_test)
