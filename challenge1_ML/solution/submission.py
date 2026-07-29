import argparse
import os
from model import Model

import pandas as pd


def get_train_data(input_dir):
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    return X_train, y_train


def get_test_data(input_dir):
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    return X_test


def main():
    # CodaBench-style scripts receive input/output folders as CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print('Reading Data')
    X_train, y_train = get_train_data(input_dir)
    X_test = get_test_data(input_dir)
    print('Starting')

    # Train the selected ML pipeline (implemented in model.py).
    m = Model()
    print('Training Model')
    m.fit(X_train, y_train)

    # Predict ages for X_test, then write the expected submission file.
    print('Running Prediction')
    prediction = m.predict(X_test)
    df = pd.DataFrame(prediction, columns=['age'])
    df.to_csv(os.path.join(output_dir, 'y_pred.csv'), index=False)


if __name__ == '__main__':
    main()
