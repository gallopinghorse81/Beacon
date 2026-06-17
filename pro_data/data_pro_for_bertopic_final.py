# -*- coding: utf-8 -*-
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
import time
from bertopic import BERTopic
from sklearn.cluster import KMeans
from umap import UMAP
from bertopic.vectorizers import ClassTfidfTransformer


BASE_DATA_PATH = "../dataset"

MODEL_PATH = '../dataset/all-mpnet-base-v2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_or_load_embeddings(dataset_name: str):


    docs_path = os.path.join(BASE_DATA_PATH, f"{dataset_name}_data", "data67500", "train",
                             "review_list_for_bertopic.npy")
    save_dir = os.path.dirname(docs_path)
    embeddings_path = os.path.join(save_dir, f"{dataset_name}_embeddings.npy")
    docs = np.load(docs_path, allow_pickle=True)

    if os.path.exists(embeddings_path):
        print(f"找到预计算的嵌入文件 '{embeddings_path}'，直接加载。")
        embeddings = np.load(embeddings_path)
    else:
        print(f"未找到嵌入文件 '{embeddings_path}'，开始生成...")

        embedding_model = SentenceTransformer(MODEL_PATH, device=DEVICE)
        embeddings = embedding_model.encode(
            docs,
            show_progress_bar=True,
            batch_size=256
        )
        print(f"嵌入向量生成完毕，形状为: {embeddings.shape}")
        np.save(embeddings_path, embeddings)
        print(f"嵌入向量已保存到: {embeddings_path}")

    print(f"成功加载 {len(docs)} 个文档和形状为 {embeddings.shape} 的嵌入向量。\n")
    return docs, embeddings, save_dir


def run_bertopic_and_save_embeddings(docs, embeddings, dataset_name: str, num_aspects: int, save_dir: str):
    print(f"--- [步骤 2/2] 为 '{dataset_name}' 运行BERTopic，目标方面数: {num_aspects} ---")
    start_time = time.time()


    embedding_model = SentenceTransformer(MODEL_PATH)
    umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    cluster_model = KMeans(n_clusters=num_aspects, random_state=42, n_init='auto')
    ctfidf_model = ClassTfidfTransformer()


    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="english",
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        ctfidf_model=ctfidf_model,
        verbose=True
    )
    print("开始使用预计算的嵌入训练BERTopic模型...")
    topic_model.fit_transform(docs, embeddings)
    topic_embeddings = topic_model.topic_embeddings_
    output_path = os.path.join(save_dir, f"{dataset_name}_topic_embeddings_{num_aspects}.npy")
    print(f"\n主题嵌入的形状: {topic_embeddings.shape}")
    np.save(output_path, topic_embeddings)
    print(f"主题嵌入已成功保存到: {output_path}")
    end_time = time.time()
    print(f"BERTopic分析完成，耗时: {end_time - start_time:.4f}s")





if __name__ == "__main__":
    DATASET_NAME = "Amazon_Instant_Video"


    NUM_ASPECTS = 1
    documents, doc_embeddings, save_directory = generate_or_load_embeddings(dataset_name=DATASET_NAME)
    run_bertopic_and_save_embeddings(
        docs=documents,
        embeddings=doc_embeddings,
        dataset_name=DATASET_NAME,
        num_aspects=NUM_ASPECTS,
        save_dir=save_directory
    )
    print("\n" + "=" * 50 + "\n")


    NUM_ASPECTS = 3
    documents, doc_embeddings, save_directory = generate_or_load_embeddings(dataset_name=DATASET_NAME)
    run_bertopic_and_save_embeddings(
        docs=documents,
        embeddings=doc_embeddings,
        dataset_name=DATASET_NAME,
        num_aspects=NUM_ASPECTS,
        save_dir=save_directory
    )
    print("\n" + "=" * 50 + "\n")


    NUM_ASPECTS = 5
    documents, doc_embeddings, save_directory = generate_or_load_embeddings(dataset_name=DATASET_NAME)
    run_bertopic_and_save_embeddings(
        docs=documents,
        embeddings=doc_embeddings,
        dataset_name=DATASET_NAME,
        num_aspects=NUM_ASPECTS,
        save_dir=save_directory
    )
    print("\n" + "=" * 50 + "\n")


    NUM_ASPECTS = 7
    documents, doc_embeddings, save_directory = generate_or_load_embeddings(dataset_name=DATASET_NAME)
    run_bertopic_and_save_embeddings(
        docs=documents,
        embeddings=doc_embeddings,
        dataset_name=DATASET_NAME,
        num_aspects=NUM_ASPECTS,
        save_dir=save_directory
    )
    print("\n" + "=" * 50 + "\n")
    print("所有配置的任务已完成。")

