import requests
import re
from constants import GITHUB_TOKEN
import time
import subprocess
import pandas as pd
import os

data = []
repo = "NETBEANS"
urls = ["https://api.github.com/repos/apache/netbeans/releases", "https://api.github.com/repos/apache/netbeans/releases?page=2"]

for url in urls:
    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f"token {GITHUB_TOKEN}"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        reponse_data = response.json()
    else:
        raise Exception(f"Error fetching commit from GitHub: {response.status_code}, {response.text}")

    for release in reponse_data:
        version = release['tag_name']
        content = release['body']
        content = re.sub(r"[\u2026]", "", content)
        content = re.sub(r"(?<!\\)'", r"\\'", content)
        bullet_points = re.split(r'\n\*', content)
        bullet_points = [bp.strip() for bp in bullet_points if bp.strip()]
        data.append({"version": version, "content": bullet_points})

    df = pd.DataFrame(data)
    # df["content"].tolist()
    # Save the DataFrame to a CSV file
df.to_csv(f"../../data/ReleaseNotes/{repo.title()}/release_notes_{repo.lower()}.csv", index=False)
print(f"Data saved to 'release_notes_{repo.lower()}.csv'")

    