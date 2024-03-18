"""

Data prep for analysis

This converts the manually generated mapping files which are in matrix format to
skinny format (one row per pair of variable and the mapping value) with binary mapping
values.

This also creates files holding the variables for each study and the embeddings for each variable.



"""
import pandas as pd
import numpy as np
from openai import OpenAI
import configparser

# Prep for use of Open AI
config = configparser.ConfigParser()
config.read('config.ini')
client = OpenAI(api_key = config.get('secrets', 'api_key'))

np.seterr(all='raise')





def get_embedding_text(v, dfdatadict):
    """ Given variables v, return the full text to use for that variable by augmenting with text from the data dictionary
    """
    ev = dfdatadict[dfdatadict['VARNAME'] == v]['VARDESC']
    if (len(ev) == 0):
        return v.replace('_','')   # replace, seems to help with similiraity mapping
    ev = v.replace('_','') + ' '+ ev.iloc[0]
    return ev


def gen_embedding_vector(embeddocs, model="text-embedding-3-large"):
    """ Generate embeddings for each document in embeddocs """
    res = client.embeddings.create(input = embeddocs, model=model)
    embeddings = [res.data[i].embedding for i in range(0,len(embeddocs))]
    return pd.Series(embeddings)


def prep_gen_study_embeddings(svfilenames):
    """ update the study variables to include embeddings based on data dictionary descriptions
        svfilenames[0]=study var file, [1]=data dictionary
    """
    for filenames in svfilenames:
        df = pd.read_csv(filenames[0],index_col=None)
        dfdict = pd.read_csv(filenames[1], index_col=None, encoding='latin1')
        df['embedtext'] = df.iloc[:, 0].apply(lambda x: get_embedding_text(x, dfdict))
        df['embedvals'] = gen_embedding_vector(df['embedtext'].tolist())
        df.to_csv(filenames[0], index=None)


def prep_mapmat_to_maplist(filename, vars1name, vars2name):
    """ Converts the manually curated mapping matrix from matrix format to skinny format (one row per variable pair and binary score) """
    # get the mapping matrix and convert to skinny
    df = pd.read_csv(filename)
    df = df.rename(columns={"Unnamed: 0": vars1name})
    df = df.melt(id_vars=vars1name, var_name=vars2name, value_name='value')
    df[vars1name] = df[vars1name].str.strip()
    df[vars2name] = df[vars2name].str.strip()

    # then recode values into 0 or 1 and sort
    df['value'] = df['value'].replace('.', 0).apply(lambda x: 1 if x != 0 else x)
    df = df.sort_values(by=[vars1name, vars2name], ascending=[True, True])
    return df


def prep_maplist_and_study_vars(mappingfiles):
    """  Generate the study var files and the mapping files for each pair of studies """
    for m in mappingfiles:
        df = prep_mapmat_to_maplist(f"./data/mappings/Crosstab_{m[0]}_{m[1]}.csv", m[0], m[1])
        df.to_csv(f"./data/analysis/maplist_{m[0]}_{m[1]}.csv", index=None)
        for varname in m:
            dfvar = pd.DataFrame(columns=['varname'],data=df[varname].unique())
            dfvar.to_csv(f"./data/analysis/studyvars_{varname}.csv", index=None)




prep_maplist_and_study_vars([["1450","1407"],["1740","34"],["1945","34"],["1945","1740"]])

svlist = [["./data/analysis/studyvars_1450.csv","./data/2016-1450/2016-1450-DD-DemoHealth.csv"],
          ["./data/analysis/studyvars_1407.csv","./data/2016-1407/2016-1407-DD-DemoHealth.csv"],
          ["./data/analysis/studyvars_1740.csv","./data/2016-1740/2016-1740-DD-DemoHealth.csv"],
          ["./data/analysis/studyvars_34.csv","./data/2016-34/2016-34-DD-DemoHealth.csv"],
          ["./data/analysis/studyvars_1945.csv","./data/2017-1945/2017-1945-DD-DemoHealth.csv"]]
prep_gen_study_embeddings(svlist)


