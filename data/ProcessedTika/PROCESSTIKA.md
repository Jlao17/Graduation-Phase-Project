## Run the files in the following order for the created dataset:
NOTE: Change the dataset name in the code snippets below to the dataset you are working with. \
1. Add issue attributes to the Tika dataset and combine the data. \
`LinkGenerator/BTLinkScrape/0_add_issue_attributes.py` 
2. Scrape release notes from the Tika txt file. \
`LinkGenerator/ScrapeReleaseNotes/scrape_release_notes_tika.py`
3. Add release notes to the existing dataset and process the data. \
`LinkGenerator/BTLinkScrape/1-2_splitword_diff.py`
4. Add false links for the release notes. \
`LinkGenerator/2.5_add_false_rn_links.py`
5. Preprocess (tokenize) the Diff and add training flags. \
`LinkGenerator/BTLinkScrape/3_tokenize_diff_training_flags.py`
