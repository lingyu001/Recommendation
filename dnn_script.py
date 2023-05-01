import time
import os
import pandas as pd
import numpy as np
import datetime
import faiss
from tqdm import tqdm
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
# from preprocess_sku import gen_data_set, gen_model_input, gen_data_set_timesplit
# from sklearn.preprocessing import LabelEncoder
from tensorflow import config
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from model_utils import DNN, EmbeddingIndex, NoMask, PoolingLayer, l2_normalize,\
 reduce_mean, SampledSoftmaxLayer, get_item_embedding, sampledsoftmaxloss
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Flatten, Concatenate, Layer, Add
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import AveragePooling1D
# from tensorflow.keras import layers

from model_utils import DNN

if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()
else:
    K.set_learning_phase(True)


script_start_time = time.time()

"""GPU Configuration"""

print(config.list_physical_devices())

gpu_devices = config.list_physical_devices('GPU')
print(f"# of available gpu: {len(gpu_devices)}")

"""# New Data"""

# ! pip3 install --upgrade pandas==1.4.2
# ! pip list

data = pd.read_pickle("~/work/ncf/sample_data/df_input_full_20230227.sav")

print("data shape: ", data.shape)

"""# run test sample"""
# data = data.sample(40000)

print(
    f""" 
    user num: {data.user_id.nunique()}\n
    item num: {data.sku_number.nunique()}\n
    data range: {data.trans_date.min()} to {data.trans_date.max()}
    
    """
)

"""# Feature Engineering"""


sparse_features = ['sku_number', 'category_path',
                   'user_id',
                   'geo_country', 'geo_region', 'geo_city', 'geo_zipcode','platform'
                   ]

SEQ_LEN = 50
NUMBER_NEG_SAMPLE = 1 # used for generate 1:1 positive:negative samples

feature_max_idx = {}
for feature in sparse_features:
    if feature == 'sku_number':
        feature_max_idx[feature] = max(data[feature].max(), data['sku_view_sequence'].explode().max()) + 1
    else:
        feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[['user_id', 'user_id_org']].drop_duplicates('user_id')
user_profile.set_index("user_id", inplace=True)
item_profile = data[["sku_number", 'sku_number_org']].drop_duplicates('sku_number')
print("user_profile shape:\n", user_profile.shape,
      "\nitem_profile shape:\n", item_profile.shape)

"""# Train Test Split"""

test_start_date = data['trans_date'].max() - datetime.timedelta(7)
print("test_start_date:\n", test_start_date)

feature_cols = ['user_id', 'geo_country', 'geo_region', 'geo_city', 'geo_zipcode','platform',
                'sku_number', 'category_path',
                'sku_purchase_seq', 'category_path_purchase_seq', 'sku_view_sequence',
                'seq_len']
train_data, train_label = data[data['trans_date'] < test_start_date][feature_cols], data[data['trans_date'] < test_start_date][['label']]
test_data, test_label = data[data['trans_date'] >= test_start_date][feature_cols], data[data['trans_date'] >= test_start_date][['label']]
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)


"""### take overlap user in test data"""

print(f"overlap user num: {len(set(train_data['user_id'].unique()) & set(test_data['user_id'].unique()))}")

test_data = test_data[test_data['user_id'].isin(train_data['user_id'].unique())]

train_model_input = train_data.to_dict("list")
test_model_input = test_data.to_dict("list")

train_label = train_label.to_numpy().ravel()
test_label = test_label.to_numpy().ravel()

for feature in ['sku_number', 'category_path','user_id','geo_country', 'geo_region', 'geo_city', 'geo_zipcode','platform',]:
  train_model_input[feature] = np.array(train_model_input[feature])
  test_model_input[feature] = np.array(test_model_input[feature])
for feature in ['sku_purchase_seq', 'category_path_purchase_seq', 'sku_view_sequence']:
  train_model_input[feature] = pad_sequences(train_model_input[feature], maxlen=SEQ_LEN, padding='post', truncating='post', value=0)
  test_model_input[feature] = pad_sequences(test_model_input[feature], maxlen=SEQ_LEN, padding='post', truncating='post', value=0)

"""# Create Embeddings"""

train_counter = Counter(train_model_input['sku_number'])
item_count = [train_counter.get(i,0) for i in range(feature_max_idx['sku_number'])]
print(len(train_counter), type(train_counter), len(item_count))



"""### Create Feature Embeddings


"""

# item features
item_feature_columns = ['sku_number']
# User features
user_feature_columns = ['user_id',
                        'category_path',
                        'geo_country', 'geo_region', 'geo_city', 'geo_zipcode','platform',
                        'sku_purchase_seq',
                        'category_path_purchase_seq',
                        'sku_view_sequence',
                        'seq_len']

user_sparse_feature_columns = ['user_id','category_path','geo_country', 'geo_region', 'geo_city', 'geo_zipcode','platform']
user_seq_sparse_feature_columns = [
    'sku_purchase_seq',
    'category_path_purchase_seq',
    'sku_view_sequence'
]

# embedding input and output specification
embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)
embedding_dim = 32
l2_reg = 1e-6
# use a dict to collect embeddings
sparse_embedding = {}

# user embeddings
for name in user_sparse_feature_columns: 
  emb = Embedding(
      input_dim = feature_max_idx[name], 
      output_dim = 16,
      embeddings_initializer=embeddings_initializer,
      embeddings_regularizer=l2(l2_reg),
      name='sparse_' + 'emb_' + name
  )
  emb.trainable = True
  sparse_embedding[name] = emb
for name in user_seq_sparse_feature_columns:
  if name[0:3] == 'sku':
    emb = Embedding(feature_max_idx['sku_number'], embedding_dim,
                            embeddings_initializer=embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name='sparse_' + 'seq_emb_'  + name,
                            mask_zero=False) # True
    emb.trainable = True
    sparse_embedding[name] = emb
  else:
    emb = Embedding(feature_max_idx['category_path'], embedding_dim,
                        embeddings_initializer=embeddings_initializer,
                        embeddings_regularizer=l2(
                            l2_reg),
                        name='sparse_' + 'seq_emb_'  + name,
                        mask_zero=False) # True
    emb.trainable = True
    sparse_embedding[name] = emb


# item embeddings
item_feature_name = item_feature_columns[0] # only include item id 
emb = Embedding(
      input_dim = feature_max_idx[item_feature_name], 
      output_dim = embedding_dim,
      embeddings_initializer=embeddings_initializer,
      embeddings_regularizer=l2(l2_reg),
      name='sparse_' + 'emb_' + item_feature_name
  )
sparse_embedding[item_feature_name] = emb

print(f"embeding check: {sparse_embedding}") # include both user and item embedding layers

"""# Create User Inputs

## Create Inputs Tensor
"""

# user features inputs
user_features = OrderedDict()
for name in user_sparse_feature_columns:
  user_features[name] = Input(
                shape=(1,), 
                name=name, 
                dtype="int32"
                )
for name in user_seq_sparse_feature_columns:
  user_features[name] = Input(
                shape=(SEQ_LEN,), 
                name=name, 
                dtype="int32"
                )
length_name = 'hist_len'
if length_name in user_feature_columns:
  user_features[length_name] = Input((1,), name=length_name, dtype='int32')

user_inputs_list = list(user_features.values())

# item feature inputs
item_features = OrderedDict()
for name in item_feature_columns:
  item_features[name] = Input(
                shape=(1,), 
                name=name, 
                dtype="int32"
                )
item_inputs_list = list(item_features.values())

print(f"user input list: {user_inputs_list}\nitem input list: {item_inputs_list}")


"""## Embed Inputs"""

# Embed sparse inputs
user_sparse_embeded_input = defaultdict(list)

for name in user_sparse_feature_columns:
  user_sparse_embeded_input[name] = sparse_embedding[name](user_features[name])

# Embed the sequence sparse inputs, need to add pooling layer to it
seq_embeded_dict = defaultdict(list)

for name in user_seq_sparse_feature_columns:
  seq_embeded_dict[name] = sparse_embedding[name](user_features[name])

# from deepctr.layers.sequence import SequencePoolingLayer
# for name in user_seq_sparse_feature_columns:
#   user_seq_sparse_embeded_input[name] = SequencePoolingLayer('mean', supports_masking=False)([seq_embeded_dict[name], user_features[length_name]])

user_seq_sparse_embeded_input = defaultdict(list)
for name in user_seq_sparse_feature_columns:
  user_seq_sparse_embeded_input[name] = AveragePooling1D(pool_size=50, padding='valid')(seq_embeded_dict[name])

# combine embeded inputs into a list
user_sparse_embedding_list = list(user_sparse_embeded_input.values()) + list(user_seq_sparse_embeded_input.values())
print(f"embeded inputs check:\n {user_sparse_embedding_list}")

"""## Combine Embeded Input (User only)"""


user_dnn_input = Flatten()(Concatenate(axis=2)(user_sparse_embedding_list))
print(f"user layer input: {user_dnn_input}")

"""# Sepecify Model Layers

### User Layers
"""


user_dnn_hidden_units = (128,64, embedding_dim)
dnn_activation='relu'
dnn_use_bn=False
l2_reg_dnn=0
l2_reg_embedding=1e-6
dnn_dropout=0
output_activation='linear'
temperature=0.05,
# sampler_config=sampler_config
seed=1024


dnn_layers = DNN(hidden_units=user_dnn_hidden_units, 
                  activation='relu', 
                  l2_reg=0, 
                  dropout_rate=0, 
                  use_bn=False, 
                  output_activation='linear',
                  seed=1024
                  )

user_dnn_output = dnn_layers(user_dnn_input)
user_dnn_output = l2_normalize(user_dnn_output)

print(f"user layer output: {user_dnn_output}")


"""### Item Layer"""

# Create item index tensor
item_vocabulary_size = feature_max_idx[item_feature_name]
item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])

# embed the item index input
item_embeding = sparse_embedding[item_feature_name]
item_embedding_weight = NoMask()(item_embeding(item_index))
pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])
pooling_item_embedding_weight = l2_normalize(pooling_item_embedding_weight)
print("pooling_item_embedding_weight:\n", pooling_item_embedding_weight)

"""### Output Layer"""

output = SampledSoftmaxLayer()(
    [pooling_item_embedding_weight, user_dnn_output, item_features[item_feature_name]])

"""## Build Model"""


model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

model.__setattr__("user_input", user_inputs_list)
model.__setattr__("user_embedding", user_dnn_output)

model.__setattr__("item_input", item_inputs_list)
model.__setattr__("item_embedding",
                  get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

"""# Model Compile and Training"""

model.compile(optimizer="adam", loss=sampledsoftmaxloss)

history = model.fit(train_model_input, train_label,
                    batch_size=512, epochs=20, verbose=1, validation_split=0.0, )

"""# Predict: Generate Item and User Embeddings"""

# 4. Generate user features for testing and full item features for retrieval
test_user_model_input = test_model_input
all_item_model_input = {"sku_number": item_profile['sku_number'].values,}

user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

print("user_embs.shape:\n", user_embs.shape)
print("item_embs.shape:\n", item_embs.shape)

"""# Recommendation"""

def recall_N(y_true, y_pred, N=30):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)

def precision_N(y_true, y_pred, N=30):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / N

test_true_label = test_data.groupby('user_id')['sku_number'].apply(list).to_dict()



index = faiss.IndexFlatIP(embedding_dim)
index.add(item_embs)
D, I = index.search(np.ascontiguousarray(user_embs), 50)
recall = []
precision = []
f1 = []
hit = 0
pred_label = {}
# for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
for i, uid in enumerate(test_user_model_input['user_id']):
    try:
        pred = [item_profile['sku_number'].values[x] for x in I[i]]
        filter_item = None
        recall_score = recall_N(test_true_label[uid], pred, N=30)
        precision_score = precision_N(test_true_label[uid], pred, N=30)
        recall.append(recall_score)
        precision.append(precision_score)
        f1_score = 2 * (precision_score * recall_score) /(precision_score + recall_score)
        f1.append(f1_score)
        pred_label[uid] = pred
        if test_true_label[uid] in pred:
            hit += 1
    except:
        pass

print("recall", np.mean(recall))
print("precision", np.mean(precision))
print("f1", np.mean(f1))
print("hit rate", hit / len(test_user_model_input['user_id']))

print("recall", np.mean(recall)*100)
print("precision", np.mean(precision)*100)
print("f1", np.mean(f1)*100)

""" save results to dir"""

# Get today's date and time as a string
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a directory with today's date and time as the name
# dir_name = f"/output/result_{now}"
# os.makedirs(dir_name)
path = f"{os.getenv('HOME')}/work/ncf/output/result_{now}"
Path(path).mkdir(parents=True)
# Open a file named 'example.txt' for writing
with open(f'{path}/result.txt', 'w') as f:
    # Write some text to the file
    f.write(f'"recall: "{np.mean(recall)*100} %\n')
    f.write(f'"precision: "{np.mean(precision)*100} %\n')
    f.write(f'"f1: "{np.mean(f1)*100} %\n')
    # Close the file
    f.close()

# Print a message to confirm that the file was written
print('File written successfully.')

print(f'pipeline end, it takes: {(time.time() - script_start_time)//60} min, ')