# EHLC_Data_Harmonization
Repository for holding content from the data harmonization use case.  This repository holds:

Crosstab_X_Y.xlsx:  manually generated mappings of variables from studies X and Y.  Overview tab in each
spreadsheet describes the tabs and details.

MappingResults.xlsx: holds the counts of manual mappings between pairs of studies.

data_prep.py: code to prep for analysis, including converting manually mapped data into analysis format
and to generate embeddings for each variable to support analysis.  A OpenAI key is needed to run this code.

run_analysis.py: code to generate a computer based mapping of variables between studies (using embeddings
for similiarity between pairs of variables) and compare the computer and manual mappings.

gen_heatmap.py: throw away code used to create a heatmap of cross study mappings based on embeddings as
a sanity check on using the embeddings.

overall_best_results.xlsx: holds results of comparing manual and computer based mappings using the OpenAI 
text-embedding-3-large model.




