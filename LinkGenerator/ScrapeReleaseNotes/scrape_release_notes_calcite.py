import requests
import pandas as pd
from bs4 import BeautifulSoup

# Define the URL of the release notes page
url = "https://calcite.apache.org/docs/history.html"  # Replace with the actual URL

def get_release_notes(url, name):
# Fetch the HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Initialize a list to hold the structured data
    data = []

    # Find the div with the class "unit four-fifths"
    release_notes_div = soup.find("div", class_="unit four-fifths")

    # Loop through all <h2> tags within the release notes div
    for h2 in release_notes_div.find_all("h2"):
        version_tag = h2.find("a")

        if version_tag:
            version = version_tag.get_text(strip=True)  # Extract version

            # Get the full text of the <h2> and split it to find the date
            h2_text = h2.get_text(strip=True)
            parts = h2_text.split("/")

            if len(parts) > 1:
                date_text = parts[1]  # Get the date part
            else:
                date_text = "N/A"  # Default value if no date found
        else:
            continue  # Skip if <a> is not found

        content = []

        # Get the next siblings (content below the <h2>)
        for sibling in h2.find_next_siblings():
            if sibling.name == "h2":  # Stop if we reach the next <h2>
                break
            # Append the text from <p> or <ul> elements to the content list
            elif sibling.name == "ul":
                # Format the <ul> items as bullet points
                for li in sibling.find_all("li"):
                    content.append(f"{li.get_text(strip=True)}")  # Bullet point
        # Append the extracted data to the list
        data.append({"version": version, "date": date_text, "content": content})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df["content"].tolist()
    # Save the DataFrame to a CSV file
    df.to_csv(f"../data/ReleaseNotes/release_notes_{name}.csv", index=False)
    print(f"Data saved to 'release_notes_{name}.csv'")

def main():
    # Call the function with the URL and name
    get_release_notes("https://calcite.apache.org/docs/history.html", "calcite")
    get_release_notes("https://calcite.apache.org/avatica/docs/go_history", "calcite_avatica_go")
    get_release_notes("https://calcite.apache.org/avatica/docs/history", "calcite_avatica")

if __name__ == "__main__":
    main()