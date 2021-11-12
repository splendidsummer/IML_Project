from dataset.data_preprocessing_category import *
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    dict_path = 'arff_paths_dict.json'
    dataset_name = 'splice'
    with open(dict_path, 'r') as f:
        arff_path_dict = json.load(f)
    splice_data, splice_meta = load_category_data(arff_path_dict, dataset_name)
    meta_attribute_dict = get_meta_attributes(splice_meta, dataset_name)
    # print(len(meta_attribute_dict.keys()))
    splice_df = pd.DataFrame(splice_data)
    df_attribute_dict = get_df_attributes(splice_df)

    for meta_key, df_key in zip(meta_attribute_dict.keys(), df_attribute_dict.keys()):
        assert len(meta_attribute_dict[meta_key]) == len(df_attribute_dict[df_key])
        # print('No {} value get out the defined value range by meta'.format(meta_key))

    with open('meta_attribute_dict.pkl', 'wb') as f:
        pickle.dump(meta_attribute_dict, f)

    with open('df_attribute_dict.pkl', 'wb') as f:
        pickle.dump(df_attribute_dict, f)

    # LabelEncoder here to avoid  sparse input data
    le = LabelEncoder()
    categorize_method = le.fit_transform
    splice_df = splice_df.apply(categorize_method)
    # print(type(splice_df))
    encoded_unique_values = {}

    for col_name in splice_df.columns:
        encoded_unique_values[col_name] = splice_df[col_name].unique()

    with open('encoded_unique_values.pkl', 'wb') as f:
        pickle.dump(encoded_unique_values, f)

    splice_df_unique= splice_df.drop_duplicates()
    print('{} duplicate samples get deleted'.format(len(splice_df.index) - len(splice_df_unique.index)))

    splice_array = splice_df_unique.values

    with open('splice_array.pkl', 'wb') as f:
        pickle.dump(splice_array, f)

    splice_df_unique.to_csv('splice_df.csv')
















