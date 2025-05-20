## Run the files in the following order for the created dataset:
1. Process the commit and issue text. \
`LinkGenerator/1_splitword.py` 
2. Scrape release notes from the Calcite website. \
`LinkGenerator/ScrapeReleaseNotes/scrape_release_notes_calcite.py`
3. Add release notes to the existing dataset. \
`LinkGenerator/AddReleaseNotesToData/1.5_add_release_notes_calcite.py` 
4. Process the Diff. \
`LinkGenerator/UpdateDiff/2_get_diff_calcite.py`
5. Add false links for the release notes. \
`LinkGenerator/2.5_add_false_rn_links.py`
6. Preprocess (tokenize) the Diff. \
`LinkGenerator/UpdateDiff/3_tokenize_diff.py`