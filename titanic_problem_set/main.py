from preliminary_inspection import load_data, find_missing_data, find_most_frequent, find_uniques
from univariate_analysis import concat_train_test, set_family_size, set_age_interval, set_fare_interval,create_sex_Pclass,  parse_names, process_name

TRAIN_PATH = "titanic_problem_set/data/train.csv"
TEST_PATH = "titanic_problem_set/data/test.csv"


def main():
    # Load train and test data
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    # Sanity check
    train_df.head()
    test_df.head()
    train_df.info()
    test_df.info()
    train_df.describe()
    test_df.describe()

    # Preliminary inspection
    df_missing_train = find_missing_data(train_df)
    df_missing_test = find_missing_data(test_df)
    df_most_frequent_train = find_most_frequent(train_df)
    df_most_frequent_test = find_most_frequent(test_df)
    df_train_unique = find_uniques(train_df)
    df_test_unique = find_uniques(test_df)

    # Univaraite analysis
    all_df = concat_train_test(train_df, test_df)
    all_df = set_family_size(all_df)
    all_df = set_age_interval(all_df)
    all_df = set_fare_interval(all_df)
    all_df = create_sex_Pclass(all_df)
    all_df = process_name(all_df)
    print(all_df.head(3))


if __name__ == "__main__":
    main()