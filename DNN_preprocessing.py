import numpy as np
import pandas as pd
import datetime
from db_connectors.load import bigquery_reader

# func to get data
def get_sale_view_data(bq_project_id, json_key_path, days, top_user_num=None):
    if top_user_num:
        query = f"""
    
        -- combine mik_sales with view data
        -- just pick top 200 user first
        WITH cte1 AS (
          SELECT
            user_id,
            COUNT(trans_date) AS num_trans
          FROM `Data_Infra_Eng.mik_sales`
          WHERE data_source = "MIK"
            AND DATE_DIFF(CURRENT_DATE(), trans_date, DAY) < {days}
          GROUP BY user_id
          ORDER BY num_trans DESC
          LIMIT {top_user_num}
        ), cte2 AS (
    
          SELECT 
            t1.user_id, 
            t1.sku_number, 
            t1.qty, 
            t1.trans_date,
            t1.created_time,
            t1.data_source,
            t2.full_taxonomy_path as category_path,
            ARRAY_AGG(t1.sku_number) --IFNULL(t2.sku_number, "na")
              OVER (
                PARTITION BY t1.user_id 
                ORDER BY t1.trans_date, t1.created_time ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
              ) AS sku_purchase_seq,
            ARRAY_AGG(IFNULL(t2.full_taxonomy_path, "na")) 
              OVER (
                PARTITION BY t1.user_id 
                ORDER BY t1.trans_date, t1.created_time ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
              ) AS category_path_purchase_seq
          FROM `Data_Infra_Eng.mik_sales` t1
          LEFT JOIN `Data_Infra_Eng.mik_item` t2
            ON t1.sku_number = t2.sku_number
          WHERE data_source = "MIK"
            AND DATE_DIFF(CURRENT_DATE(), trans_date, DAY) < {days} 
            AND t1.user_id IN (SELECT user_id FROM cte1)
          ORDER BY user_id, trans_date, created_time ASC
        )
        SELECT
          *
        FROM cte2 t1
        LEFT JOIN `Data_Infra_Eng.user_behavior` t2
          ON CAST(t1.user_id AS STRING) = t2.user_id
        ;
        """
    else:
        query = f"""


        -- combine mik_sales with view data

        WITH cte2 AS (

          SELECT 
            t1.user_id, 
            t1.sku_number, 
            t1.qty, 
            t1.trans_date,
            t1.created_time,
            t1.data_source,
            t2.full_taxonomy_path as category_path,
            ARRAY_AGG(t1.sku_number) --IFNULL(t2.sku_number, "na")
              OVER (
                PARTITION BY t1.user_id 
                ORDER BY t1.trans_date, t1.created_time ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
              ) AS sku_purchase_seq,
            ARRAY_AGG(IFNULL(t2.full_taxonomy_path, "na")) 
              OVER (
                PARTITION BY t1.user_id 
                ORDER BY t1.trans_date, t1.created_time ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
              ) AS category_path_purchase_seq
          FROM `Data_Infra_Eng.mik_sales` t1
          LEFT JOIN `Data_Infra_Eng.mik_item` t2
            ON t1.sku_number = t2.sku_number
          WHERE data_source = "MIK"
            AND trans_date BETWEEN '2023-01-24' AND '2023-02-27'
          ORDER BY user_id, trans_date, created_time ASC
        ), cte3 AS (
        SELECT
          user_id,
          APPROX_TOP_COUNT(geo_country, 1) AS geo_country,
          APPROX_TOP_COUNT(geo_region, 1) AS geo_region,
          APPROX_TOP_COUNT(geo_city, 1) AS geo_city,
          APPROX_TOP_COUNT(geo_zipcode, 1) AS geo_zipcode,
          APPROX_TOP_COUNT(platform, 1) AS platform,
        FROM`atomic.events`
        WHERE derived_tstamp BETWEEN '2023-01-24' AND '2023-02-27'
        GROUP BY user_id  

        )

        SELECT
          cte2.*,
          cte3.geo_country,
          cte3.geo_region,
          cte3.geo_city,
          cte3.geo_zipcode,
          cte3.platform,
          t.*
        FROM cte2
        LEFT JOIN cte3
          ON CAST(cte2.user_id AS STRING) = cte3.user_id
        LEFT JOIN `Data_Infra_Eng.user_behavior` t
          ON CAST(cte2.user_id AS STRING)= t.user_id

        """
    df = bigquery_reader(
            project_id=bq_project_id, json_credentials_path=json_key_path,
            query_string=query
        )
    df = df[df['user_id_1'].notna()].reset_index(drop=True)
    df['sku_view_sequence'] = df['user_behavior'].apply(lambda x: np.array([y['item'] for y in x]))
    print(f"""
            df shape: {df.shape}, user number: {df['user_id'].nunique()}, 
            avg trans per user: {df.groupby('user_id').size().mean()}, 
            item number: {df['sku_number'].nunique()}
            """)
    return df
# func to create negative sample
def create_negative_sample(data, candidate_item, negsample_ratio):
    # get positive sample data
    df_pos = data.copy()
    df_pos['label'] = 1

    # create negative data
    df_neg = pd.concat([df_pos.copy()] * negsample_ratio, ignore_index=True)
    df_neg['label'] = 0
    # negative sampling
    neg_sku = candidate_item[~candidate_item['sku_number'].isin(df_pos['sku_number'])] \
        .sample(df_pos.shape[0] * negsample_ratio)
    df_neg['sku_number'] = neg_sku['sku_number'].values
    df_neg['category_path'] = neg_sku['category_path'].values

    return pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)

# func to generate input data
def gen_input_data(df, negsample_ratio, seq_len):
    # get candidate items and categories
    candidate_sku = df[['sku_number','category_path']].drop_duplicates()
    # group by user_id
    user_group = df.groupby('user_id')
    # negative sampling for each user group
    df_res = []
    for user_id, data in user_group:
        # first refine sequence len to < seq_len
        for col in ['sku_purchase_seq','category_path_purchase_seq','sku_view_sequence']:
            data[col] =  data[col].apply(lambda x: x[0:seq_len])
#             data[col] = data[col].apply(lambda x: np.lib.pad(x,
#                                                              (seq_len - x.shape[0],0),
#                                                              'constant',
#                                                              constant_values=('na')))
        # create negative sample and combine with positive sample
        df_sp = create_negative_sample(data = data, candidate_item = candidate_sku, negsample_ratio = negsample_ratio)
        df_res.append(df_sp)
    df_res = pd.concat(df_res).reset_index(drop=True)
    df_res['seq_len'] = seq_len
    return df_res[[
        'user_id', 'sku_number', 'category_path', 'trans_date', 'created_time',
        'sku_purchase_seq', 'category_path_purchase_seq', 'sku_view_sequence', 'seq_len',
        'label'
    ]]

# func to encode item id
def label_transform(lbe, x):
    try:
        return lbe.transform(x) + 1
    except:
        return np.array([])


def encode_features(df_input):
    from sklearn.preprocessing import LabelEncoder

    # store original item id and user id
    df_input['sku_number_org'] = df_input['sku_number']
    df_input['user_id_org'] = df_input['user_id']

    # specify features and sequence features
    sparse_features = [
        'sku_number', 'category_path',
        'user_id', 'geo_country', 'geo_region',
        'geo_city', 'geo_zipcode', 'platform'
    ]
    seq_sparse_feature = ['sku_purchase_seq', 'category_path_purchase_seq', 'sku_view_sequence']

    # get full set of item and category
    full_item_set = np.append(
        np.unique(
            np.concatenate(
                (
                    df_input['sku_number'].values,  # all sku in sales
                    df_input['sku_view_sequence'].explode().values  # all sku in views
                )
            )
        )
        , 'na')
    full_cat_set = np.append(df_input['category_path'].unique(), 'na')

    # encode item and cat set
    lbe_sku = LabelEncoder()
    lbe_sku.fit(full_item_set)
    lbe_cat = LabelEncoder()
    lbe_cat.fit(full_cat_set)
    encoded_item_set = lbe_sku.transform(full_item_set) + 1  # +1 to remove 0, 0 leave it for missing value
    encoded_cat_set = lbe_cat.transform(full_cat_set) + 1  # +1 to remove 0, 0 leave it for missing value

    # create encode dict
    sku_dict = {full_item_set[i]: encoded_item_set[i] for i in range(len(full_item_set))}
    cat_dict = {full_cat_set[i]: encoded_cat_set[i] for i in range(len(full_cat_set))}

    # fit and transform sparse features
    for feature in sparse_features:
        # need to store sku encoder
        if feature == 'sku_number':
            df_input[feature] = lbe_sku.transform(df_input[feature]) + 1  # add one to all the encoded categories labels
        # need to store
        elif feature == 'category_path':
            df_input[feature] = lbe_cat.transform(df_input[feature]) + 1  # add one to all the encoded categories labels
        else:
            lbe = LabelEncoder()
            df_input[feature] = lbe.fit_transform(
                df_input[feature]) + 1  # add one to all the encoded categories labels

    # encode sequence features
    for feature in seq_sparse_feature:
        if feature == 'sku_purchase_seq' or feature == 'sku_view_sequence':
            df_input[feature] = df_input[feature].apply(lambda x: np.array([sku_dict[c] for c in x]))
        elif feature == 'category_path_purchase_seq':
            df_input[feature] = df_input[feature].apply(lambda x: np.array([cat_dict[c] for c in x]))

    # get feature index table
    feature_max_idx = {}
    for feature in sparse_features:
        if feature == 'sku_number':
            feature_max_idx[feature] = encoded_item_set.max() + 1
        elif feature == 'category_path':
            feature_max_idx[feature] = encoded_cat_set.max() + 1
        else:
            feature_max_idx[feature] = df_input[feature].max() + 1  # plus one to the max

    return df_input, feature_max_idx
# Get data 35 days
days = 35
top_user_num = 200
df = get_sale_view_data(days, top_user_num)

# Negative sampling
candidate_sku = df[['sku_number','category_path']].drop_duplicates()
negsample_ratio = 1
SEQ_LEN = 50

df_input = gen_input_data(df, negsample_ratio, SEQ_LEN)
print(df_input.shape)
df_input, feature_max_idx = encode_features(df_input)
print(df_input.shape, feature_max_idx)

df_input.to_pickle('df_input_full.sav')