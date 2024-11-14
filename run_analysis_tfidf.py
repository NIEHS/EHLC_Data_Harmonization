"""

Generate computer based mappings of variables from pairs of studies and compare with
human generated mappings.

Compare compute and human mappings using precision, recall, and F1 scores.

Relies on output from data_prep.py

"""
import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

def gen_pred_data(study1, study2):
    """ Calculate the predicted var-to-var mappings by using distance between embeddings for each variable """
    # first compute the similiarity matrix from the per study variables (from data_prep.py)
    df1 = pd.read_csv(f'./data/analysis/studyvars_{study1}.csv', usecols=['varname','embedtext'])
    df2 = pd.read_csv(f'./data/analysis/studyvars_{study2}.csv', usecols=['varname', 'embedtext'])
    combined_list = df1['embedtext'].tolist() + df2['embedtext'].tolist()

    # Initialize an instance of tf-idf Vectorizer, not being efficient here, but easier to debug and get into the same format as done with the embedding models
    lv = TfidfVectorizer()
    lm = lv.fit_transform(combined_list).toarray()
    res = []
    for idf1 in range(0, len(df1)):
        for idf2 in range(0, len(df2)):
            sim = float(cosine_similarity([lm[idf1]], [lm[len(df1) + idf2]])[0][0])
            res.append([df1['varname'][idf1], df2['varname'][idf2], sim])

    # then convert to skinny dataframe to match the manual mapping file
    df = pd.DataFrame(data = res, columns=[study1, study2, 'predvalue'])
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
    dfpred.to_csv(f"./data/analysis/tmp_{study1}_{study2}.csv")

    # loop over different thresholds for converting computer generated similiarity scores to binary
    for t in np.arange(0.2, 0.8, 0.05):
        dfpred['predvalue_binary'] = (dfpred['predvalue'] >= t).astype(int)
        precision, recall, f1, tp, fp, fn = compute_performance(dftrue, dfpred)
        dfres = pd.concat([pd.DataFrame([[study1, study2, t, precision, recall, f1, tp, fp, fn]], columns=dfres.columns), dfres], ignore_index=True)

# sort and output the results
dfres.sort_values(by=['study1', 'study2', 'f1'])
dfres.to_csv(f"./data/analysis/overall_results.tfidf.csv", index=None)

# then output the max results across thresholds for each study pair
idx = dfres.groupby(['study1', 'study2'])['f1'].idxmax()
df_max_rows = dfres.loc[idx]
df_max_rows.to_csv(f"./data/analysis/overall_best_results.tfidf.csv", index=None)


