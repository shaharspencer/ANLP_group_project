import pandas as pd
from docopt import docopt
from sklearn.model_selection import train_test_split

usage = '''
split_train_test CLI.
provide a comma-seperated list of files to process.
Usage:
    split_train_test.py <files_to_process> 
'''

TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.1

def split_train_test(csv_source_file: str):
    df = pd.read_csv(csv_source_file, encoding='utf-8')
    # split dataset into validation and train, and test
    train_and_validation, test = train_test_split(df,
                                                  test_size=TEST_SIZE,
                                                  random_state=1)
    train, validation = train_test_split(train_and_validation,
                                         test_size=(VALIDATION_SIZE /
                                                    (VALIDATION_SIZE +
                                                     TRAIN_SIZE)),
                                         random_state=1)

    train_csv_name = csv_source_file.replace(".csv", "") + "train_split.csv"
    train.to_csv(train_csv_name, encoding='utf-8')
    validation_csv_name = csv_source_file.replace(".csv", "") + "validation_split.csv"
    validation.to_csv(validation_csv_name, encoding='utf-8')
    test_csv_name = csv_source_file.replace(".csv", "") + "test_split.csv"
    test.to_csv(test_csv_name, encoding='utf-8')

    return train, validation, test


if __name__ == '__main__':
    args = docopt(usage)

    file_list = args["<files_to_process"].split(",")

    for file in file_list:
        split_train_test(file)

    file_to_process = args["<file_to_process>"]



