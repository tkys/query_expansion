import numpy as np
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import time


def load_embeddings(embedding_path):
    # 単語埋め込みモデルを読み込む
    return KeyedVectors.load_word2vec_format(embedding_path, binary=True)

def get_top_k_words(model, query_word, top_k):
    # 与えられた単語に最も近い単語を取得する
    return model.most_similar(query_word, topn=top_k)

def cluster_words(words, n_clusters):
    # 単語のクラスタリングを行う
    word_vectors = np.array([model[word] for word, _ in words])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(word_vectors)
    return kmeans.labels_

# モデルとクエリを設定

embedding_path = './entity_vector/entity_vector.model.bin' # 単語埋め込みモデルのパス
model = load_embeddings(embedding_path)


query_word = '電池' # クエリ単語
top_k = 50 # 取得する単語数
n_clusters = 3 # クラスタの数



# トップkの単語を取得
top_k_words = get_top_k_words(model, query_word, top_k)
print(f"top_k_words:{top_k_words}")


# クラスタリング
clusters = cluster_words(top_k_words, n_clusters)



st = time.time()

# クラスタごとの単語を表示
for i in range(n_clusters):
    print(f"Cluster {i}: {[word for j, (word, _) in enumerate(top_k_words) if clusters[j] == i]}")

print(f"time_clustering_spent:{round(time.time()-st,8)}")


