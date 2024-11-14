import faiss
import numpy as np
import os
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 調用模型
pipe = pipeline("feature-extraction", model="BAAI/bge-large-zh-v1.5")

# 輸入文本
docs = [["貓咪在喝水、", "小狗在跑步、"], 
        ["貓咪在吃飯、", "小狗在玩球、"],
        ["貓咪在搞怪、", "小狗在吐舌、"]]

# 文字向量: [doc size, sequence length, word length, embeddings size]
features = pipe(docs)

# 句子向量: [doc size, sequence length, embeddings size]
docs_embeddings = np.array(features).mean(axis=2)
vector_dim = docs_embeddings.shape[2]

faiss_index_file = "faiss.bin"
mapping_file = "mapping.txt"

# 如果無索引就創建新的，有的話就用舊的
if not os.path.exists(faiss_index_file):
    # 創建新的 FAISS 索引
    index = faiss.IndexFlatL2(vector_dim)
    # 創建新的映射文件
    with open(mapping_file, "w") as f:
        mapping = []
else:
    # 讀取現有的 FAISS 索引
    index = faiss.read_index(faiss_index_file)
    # 讀取現有的映射文件
    with open(mapping_file, "r") as f:
        mapping = [line.strip() for line in f]

# 添加數據到索引並更新映射
for i, doc in enumerate(docs_embeddings):
    index.add(doc)
    for seq in range(len(docs[i])):
        mapping.append(f"{i},{seq}")

# 更新索引
faiss.write_index(index, faiss_index_file)

# 更新映射文件
with open(mapping_file, "w") as f:
    for line in mapping:
        f.write(line + "\n")

# 查詢向量
query = "貓咪"
query_feature = pipe(query)
query_embedding = np.array(query_feature).mean(axis=1)

# 查詢最近的向量
k = 3
distances, indices = index.search(query_embedding, k)

# 查詢對應的句子
retrieved_docs = []
for i in range(k):
    doc_idx, seq_idx = map(int, mapping[indices[0][i]].split(','))
    retrieved_docs.append(docs[doc_idx][seq_idx])
    print("最相似的句子:", docs[doc_idx][seq_idx])
    print("相似度:", distances[0][i])

docs_embeddings = np.reshape(np.array(docs_embeddings),[6,1024])

# 將所有向量和查詢向量合併
all_embeddings = np.vstack([docs_embeddings, query_embedding])

# 使用 PCA 降维到2D
pca = PCA(n_components=2)
all_embeddings_2d = pca.fit_transform(all_embeddings)

# 繪製向量圖
plt.figure(figsize=(5, 4))
plt.scatter(all_embeddings_2d[:-1, 0], all_embeddings_2d[:-1, 1], c='blue', label='Documents')
plt.scatter(all_embeddings_2d[-1, 0], all_embeddings_2d[-1, 1], c='red', label='Query')

# 標記查詢向量和最近的向量
for i, (x, y) in enumerate(all_embeddings_2d[:-1]):
    if i in indices[0]:
        plt.scatter(x, y, c='green', label='Retrieved' if 'Retrieved' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.legend()
plt.title("2D Visualization of Document Embeddings and Query Embedding")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
