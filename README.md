# Repository for Graduation Phase Project: Automatic Tracing of Release-Log to Code Changes​
Code based on the paper [``BTLink : automatic link recovery between issues and commits based on pre-trained BERT model``](https://link.springer.com/article/10.1007/s10664-023-10342-7).

Data can be found at: https://zenodo.org/records/16454274

The steps to preprocess each dataset can be found in ``data/Processed<dataset name>`` for each respective dataset.

``models`` folder contains both soft share and heavy share model architectures, train/test code, and util functions used during training and testing.

``util_functions.py`` contains random functions used during the thesis (e.g., combining datasets, analyzing the results, creating graphs etc.). Not all functions in this file have been applied to the final tool.

Add your own Github token as a constant in constants.py or set the variable `GITHUB_TOKEN` to your token where it is used in the code.
This token is used to access the Github API to fetch issues and commits.
