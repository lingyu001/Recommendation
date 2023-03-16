
import numpy as np
import faiss

def precision_N(y_true, y_pred, N=30):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / N

def recall_N(y_true, y_pred, N=30):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)

def get_eval_score(test_data, test_model_input, item_embs, user_embs, embedding_dim, item_profile, candidate_count):
    test_true_label = test_data.groupby('user_id')['sku_number'].apply(list).to_dict()

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embs)
    D, I = index.search(np.ascontiguousarray(user_embs), candidate_count)
    recall = []
    precision = []
    f1 = []
    hit = 0
    pred_label = {}
    for i, uid in enumerate(test_model_input['user_id']):
        try:
            pred = [item_profile['sku_number'].values[x] for x in I[i]]
            filter_item = None
            recall_score = recall_N(test_true_label[uid], pred, N=candidate_count)
            precision_score = precision_N(test_true_label[uid], pred, N=candidate_count)
            recall.append(recall_score)
            precision.append(precision_score)
            f1_score = 2 * (precision_score * recall_score) /(precision_score + recall_score)
            f1.append(f1_score)
            pred_label[uid] = pred
            if test_true_label[uid] in pred:
                hit += 1
        except:
            pass
    accuracy = hit / len(test_model_input['user_id'])

    print("recall: ", np.mean(recall))
    print("precision: ", np.mean(precision))
    print("f1: ", np.mean(f1))
    print("accuracy: ", accuracy)
    return precision, recall, f1, accuracy

