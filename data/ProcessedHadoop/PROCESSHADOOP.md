## Run the files in the following order: 
NOTE: Change the dataset name in the code snippets below to the dataset you are working with. \
1. Scrape data from the Hadoop change logs, Jira and GitHub. \
`LinkGenerator/HadoopScrape/scrape_hadoop_data.py`
2. Process the commit and issue text. \
`LinkGenerator/HadoopScrape/1-1.5_splitword_hadoop.py` 
3. Fix the commit hashes (the hashes from the pull request earlier are not in the official Hadoop repo). \
`LinkGenerator/HadoopScrape/fix_hadoop_commit_hash.py` 
4. Get the proper Diff format for the Hadoop data. \
`LinkGenerator/UpdateDiff/2_get_diff_hadoop.py`
5. Add false links for issue and commits (only true links have been scraped). \
`LinkGenerator/HadoopScrape/2.4_add_false_links_hadoop.py` 
6. Add false links for the release notes. \
`LinkGenerator/2.5_add_false_rn_links.py` 
7. Add training flags. \
`LinkGenerator/HadoopScrape/2.6_add_training_flags_hadoop.py`
8. Preprocess (tokenize) the Diff. \
`LinkGenerator/UpdateDiff/3_tokenize_diff.py`


