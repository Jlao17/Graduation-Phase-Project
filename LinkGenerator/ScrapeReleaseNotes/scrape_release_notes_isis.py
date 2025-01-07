import requests
import pandas as pd
from bs4 import BeautifulSoup

# Define the URL of the release notes page
url = "https://causeway.apache.org/versions/1.16.1/release-notes/release-notes.html"  # Replace with the actual URL

def get_release_notes(url, name):
# Fetch the HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Initialize a list to hold the structured data
    data = []

    # Find the div with the class "unit four-fifths"
    release_notes_div = soup.find("div", id="doc-content")

    # Loop through all <h2> tags within the release notes div
    for div in release_notes_div.find_all("div", class_="sect1"):
        version_tag = div.find("h2")
        version = version_tag.get_text(strip=True)

        content = []

        release_notes_content_div = div.find("div", class_="sectionbody")

        for content_div in release_notes_content_div.find_all("div", class_="sect2"):
            sect3_divs = content_div.find_all("div", class_="sect3")

            # If sect3 divs exist, extract content from them
            if sect3_divs:
                for sect3 in sect3_divs:
                    ulist = sect3.find("div", class_="ulist")
                    if ulist:
                        ulist_ul = ulist.find("ul")
                        for li in ulist_ul.find_all("li"):
                            content.append(f"{li.get_text(strip=True)}")
            else:
                # If no sect3 divs, extract content from sect2
                ulist = content_div.find("div", class_="ulist")
                if ulist:
                    ulist_ul = ulist.find("ul")
                    for li in ulist_ul.find_all("li"):
                        content.append(f"{li.get_text(strip=True)}")



        data.append({"version": version, "content": content})

        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        df["content"].tolist()
        # Save the DataFrame to a CSV file
        df.to_csv(f"../data/ReleaseNotes/Isis/release_notes_{name}.csv", index=False)
        print(f"Data saved to 'release_notes_{name}.csv'")

def main():
    # Call the function with the URL and name
    get_release_notes("https://causeway.apache.org/versions/1.16.1/release-notes/release-notes.html", "isis")
    # get_release_notes("https://calcite.apache.org/avatica/docs/go_history", "calcite_avatica_go")
    # get_release_notes("https://calcite.apache.org/avatica/docs/history", "calcite_avatica")

if __name__ == "__main__":
    main()