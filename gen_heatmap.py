"""

Code to generate a heat map of computer generated mappings
(Code copies working functions from run_analysis)

"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt
import ast



def load_study(study):
    """ Read in study and return the variables as list and embeddings as matrix """
    dfv = pd.read_csv(f"./data/analysis/studyvars_{study}.csv", index_col=None)
    vars = list(dfv['varname'])
    embeddll = dfv['embedvals'].apply(ast.literal_eval).tolist()
    embeddarr = np.array(embeddll)
    return vars, embeddarr

def compute_similarity(embeddings1, embeddings2):
    """ Calculate the similarity between embeddings """
    # Normalize embeddings
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    # Compute cosine similarity
    similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)   # helped and hurt
    similarity_matrix = np.dot(embeddings1, embeddings2.T)
    return similarity_matrix


# Compute pairwise similarities between the two sets of embeddings
vars1, embeddings1 = load_study("1740")
vars2, embeddings2 = load_study("34")
similarity_matrix = compute_similarity(embeddings1, embeddings2)

# Create a heatmap to visualize the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap="YlGnBu", annot=False, fmt=".2f", cbar=True, square=True,
            xticklabels=vars2,yticklabels=vars1)
plt.title("Document Similarity Heatmap")
plt.xlabel("Set 2 Documents")
plt.ylabel("Set 1 Documents")
plt.show()