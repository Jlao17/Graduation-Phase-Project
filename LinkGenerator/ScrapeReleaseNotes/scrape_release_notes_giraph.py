import re
import pandas as pd

# Read the release notes text file
with open("../../data/ReleaseNotes/Giraph/formatted_release_notes_giraph.txt", "r", encoding="utf8") as file:
    release_notes_text = file.readlines()



# Pattern to detect lines with release version (e.g., Release 2.7.0 - 1/31/2023)
release_version_pattern = re.compile(r'^Release \d+\.\d+\.\d+ - \d{1,2}/\d{1,2}/\d{4}$')

# List to store the extracted bullet points
bullet_points = []
current_bullet_point = ""

# Iterate over each line
for line in release_notes_text:
    # Check if the line starts with a '*' (bullet point)
    if "*" in line:

        # If we are already accumulating a bullet point, add the previous one to the list
        if current_bullet_point:
            bullet_points.append(current_bullet_point.strip())

        # Start a new bullet point, removing the leading '*' and any extra spaces
        current_bullet_point = line.lstrip('*').strip()

    # If it's not a bullet point and matches a release version pattern, skip it
    elif release_version_pattern.match(line.strip()):
        continue

    else:
        # If it's not a bullet point or a release version, add the line to the current bullet point
        current_bullet_point += " " + line.strip()

# Add the last accumulated bullet point if any
if current_bullet_point:
    bullet_points.append(current_bullet_point.strip())

print(bullet_points)
# Two regex patterns for different formats
start_pattern = r"^(GIRAPH-\d{1,4}): (.*?)(?: \((.*?)\))?$"  # Matches "TIKA-### - Content (Author)"

result_rows = []

for point in bullet_points:
    match_start = re.match(start_pattern, point)
    if match_start:
        # Handle "* TIKA-### - Content (Author)"
        tracking_id = match_start.group(1)
        content = match_start.group(2).strip()

        result = {"tracking_id": tracking_id, "content": content}

        result_rows.append(result)

# Create a DataFrame from the result rows
df = pd.DataFrame(result_rows)

# # Save the DataFrame to a CSV file
df.to_csv("../../data/ReleaseNotes/Giraph/release_notes_giraph.csv", index=False)

# Show the resulting DataFrame (optional)
print(df)
