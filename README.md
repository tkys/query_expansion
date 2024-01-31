# query_expansion
---


## 類似語のクラスタリングによって、多様なクエリ拡張単語を得る

### 狙い

 類似度が高い単語を一定数取得してそれぞれを比較すると意味も含意関係も多様に異なる。
 
 類似単語をクラスタリングし、クラスタごとから新候補の単語を選択することで、多様な意味の異なりをクエリに反映させる

--- 
-  ` topk-clustering.py`

1.word2vec学習済みのモデルから、入力クエリに対して類似単語をtop_k 数十件取得する

2.取得した単語をクラスタリングする。クラスタ数は決め打ち。

3.各クラスタから入力クエリ以外の単語を選択する（選択はランダム、クラスタ中心などが候補）

```
query_word = 'エネルギー' # クエリ単語
top_k = 30 # 取得する単語数
n_clusters = 5 # クラスタの数


>>>
Cluster 0: ['運動エネルギー', '中性子', '太陽エネルギー', '[運動エネルギー]', '[プラズマ]', '光エネルギー', '熱量', '[位置エネルギー]', '波動', '[熱エネルギー]', 'プラズマ', '[光子]', '輻射', '[潜熱]', '[反物質]', '電磁波', '[中性子線]']
Cluster 1: ['[エネルギー]', '[熱]', '熱']
Cluster 2: ['[中性子]', '粒子', '質量', '[電子]', '[電荷]', '物質']
Cluster 3: ['魔力', 'パワー']
Cluster 4: ['重力', '[重力]']

```
