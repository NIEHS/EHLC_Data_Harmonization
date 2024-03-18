"""

Generate computer based mappings of variables from pairs of studies and compare with
human generated mappings.

Compare compute and human mappings using precision, recall, and F1 scores.

Relies on output from data_prep.py

"""
import pandas as pd
import numpy as np
import ast


def load_study(study):
    """ Read in study and return the variables as list and embeddings as matrix """
    dfv = pd.read_csv(f"./data/analysis/studyvars_{study}.csv", index_col=None)
    vars = list(dfv['varname'])
    embeddll = dfv['embedvals'].apply(ast.literal_eval).tolist()
    embeddarr = np.array(embeddll)
    return vars, embeddarr

def compute_performance(df_true, df_pred):
    """ Compute performance, assumes identical sorting of columns, 1=match, 0=no_match """
    merged_df = pd.concat([df_true.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
    tp = ((merged_df['value'] == 1) & (merged_df['predvalue_binary'] == 1)).sum()
    fp = ((merged_df['value'] == 0) & (merged_df['predvalue_binary'] == 1)).sum()
    fn = ((merged_df['value'] == 1) & (merged_df['predvalue_binary'] == 0)).sum()
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall)/(precision + recall)
        return precision, recall, f1, tp, fp, fn
    except:
        return np.nan, np.nan, np.nan, tp, fp, fn


def compute_similarity(embeddings1, embeddings2):
    """ Calculate the similarity between embeddings """
    # Normalize embeddings
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    # Compute cosine similarity
    similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)   # helped and hurt
    similarity_matrix = np.dot(embeddings1, embeddings2.T)
    return similarity_matrix


def gen_pred_data(study1, study2):
    """ Calculate the predicted var-to-var mappings by using distance between embeddings for each variable """
    # first compute the similiarity matrix from the per study variables (from data_prep.py)
    vars1, embeddarr1 = load_study(study1)
    vars2, embeddarr2 = load_study(study2)
    similarity_matrix = compute_similarity(embeddarr1, embeddarr2)

    # then convert to skinny dataframe to match the manual mapping file
    df = pd.DataFrame(columns=[study1, study2, 'predvalue'])
    for i1, v1 in enumerate(vars1):
        for i2, v2 in enumerate(vars2):
            df = pd.concat([pd.DataFrame([[v1, v2, similarity_matrix[i1,i2]]], columns=df.columns), df], ignore_index=True)
    df = df.sort_values(by=[study1, study2])
    return df


# For each pair of studies, load the manual generated mapping file, generate
# # the computer based mapping file, then compare.
#
studymaps = [["1450","1407"],["1740","34"],["1945","34"],["1945","1740"]]
dfres = pd.DataFrame(columns=['study1','study2','threshold','precision','recall','f1','tp','fp','fn'])
for sm in studymaps:
    study1 = sm[0]
    study2 = sm[1]
    dftrue = pd.read_csv(f"./data/analysis/maplist_{study1}_{study2}.csv", index_col=None)
    dfpred = gen_pred_data(study1, study2)

    # loop over different thresholds for converting computer generated similiarity scores to binary
    for t in np.arange(0.2, 0.8, 0.05):
        dfpred['predvalue_binary'] = (dfpred['predvalue'] >= t).astype(int)
        precision, recall, f1, tp, fp, fn = compute_performance(dftrue, dfpred)
        dfres = pd.concat([pd.DataFrame([[study1, study2, t, precision, recall, f1, tp, fp, fn]], columns=dfres.columns), dfres], ignore_index=True)

# sort and output the results
dfres.sort_values(by=['study1', 'study2', 'f1'])
dfres.to_csv(f"./data/analysis/overall_results.csv", index=None)

# then output the max results across thresholds for each study pair
idx = dfres.groupby(['study1', 'study2'])['f1'].idxmax()
df_max_rows = dfres.loc[idx]
df_max_rows.to_csv(f"./data/analysis/overall_best_results.csv", index=None)
