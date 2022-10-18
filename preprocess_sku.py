import numpy as np
import random
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm


def gen_data_set(data, seq_max_len=50, negsample=0):
    data.sort_values("created_time", inplace=True)
    item_ids = data['sku_number'].unique()
    item_id_genres_map = dict(zip(data['sku_number'].values, data['category_path'].values))
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['sku_number'].tolist()
        genres_list = hist['category_path'].tolist()
        # rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_len = min(i, seq_max_len)
            if i != len(pos_list) - 1:
                train_set.append((
                    reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                    genres_list[i],
                    # rating_list[i]
                    ))
                for negi in range(negsample):
                    train_set.append((reviewerID, neg_list[i * negsample + negi], 0, hist[::-1][:seq_len], seq_len,
                                      genres_hist[::-1][:seq_len], item_id_genres_map[neg_list[i * negsample + negi]]))
            else:
                test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                                 genres_list[i],
                                #  rating_list[i]
                                 ))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_data_set_timesplit(data, split_time, seq_max_len=50, negsample=0):
    data.sort_values("created_time", inplace=True)
    item_ids = data['sku_number'].unique()
    item_id_genres_map = dict(zip(data['sku_number'].values, data['category_path'].values))
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['sku_number'].tolist()
        genres_list = hist['category_path'].tolist()
        time_list = hist['created_time'].tolist()
        # rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_len = min(i, seq_max_len)
            # if i != len(pos_list) - 1:
            if time_list[i] < split_time:
                train_set.append((
                    reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                    genres_list[i],
                    # rating_list[i]
                    ))
                for negi in range(negsample):
                    train_set.append((reviewerID, neg_list[i * negsample + negi], 0, hist[::-1][:seq_len], seq_len,
                                      genres_hist[::-1][:seq_len], item_id_genres_map[neg_list[i * negsample + negi]]))
            else:
                test_set.append((reviewerID, pos_list[i], 1, hist[::-1][:seq_len], seq_len, genres_hist[::-1][:seq_len],
                                 genres_list[i],
                                #  rating_list[i]
                                 ))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    train_seq = [line[3] for line in train_set]
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_genres = np.array([line[5] for line in train_set])
    train_genres = np.array([line[6] for line in train_set])
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_seq_genres_pad = pad_sequences(train_seq_genres, maxlen=seq_max_len, padding='post', truncating='post',
                                         value=0)
    train_model_input = {"user_id": train_uid, "sku_number": train_iid, "hist_sku_number": train_seq_pad,
                         "hist_category_path": train_seq_genres_pad,
                         "hist_len": train_hist_len, "category_path": train_genres}

    for key in ["geo_zip","most_frequent_device_class_general"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label

