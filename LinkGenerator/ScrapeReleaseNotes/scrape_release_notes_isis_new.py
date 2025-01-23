import requests
import pandas as pd
from bs4 import BeautifulSoup
import glob
import os

# Define the URL of the release notes page
url = "https://causeway.apache.org/versions/1.16.1/release-notes/release-notes.html"  # Replace with the actual URL


def get_release_notes(url, name):
    print("fetching: ", name)
    # Fetch the HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Initialize a list to hold the structured data
    data = []

    # Find the div with the class "unit four-fifths"
    release_notes_div = soup.find("article", class_="doc")

    # Loop through all <h2> tags within the release notes div

    version_tag = release_notes_div.find("h1")
    version = version_tag.get_text(strip=True)

    content = []

    release_notes_div = release_notes_div.find_all("div", class_="sect1")

    for div in release_notes_div:

        release_notes_content_div = div.find("div", class_="sectionbody")

        ulist = release_notes_content_div.find("div", class_="ulist")

        if ulist is not None:
            ulist_ul = ulist.find("ul")
            for li in ulist_ul.find_all("li"):
                content.append(f"{li.get_text(strip=True)}")
        else:
            continue

    data.append({"version": version, "content": content})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df["content"].tolist()
    # Save the DataFrame to a CSV file
    df.to_csv(f"../../data/ReleaseNotes/Isis/release_notes_{name}.csv", index=False)
    print(f"Data saved to 'release_notes_{name}.csv'")


def append_csv_files(input_folder, input_files_pattern, output_file):
    # Create the full path pattern for CSV files
    input_pattern = os.path.join(input_folder, input_files_pattern)

    # Get a list of all CSV files matching the pattern
    csv_files = glob.glob(input_pattern)

    # Read and append each file to a list
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all dataframes into one
    final_df = pd.concat(df_list, ignore_index=True)

    # Save the final concatenated dataframe to the output file
    final_df.to_csv(output_file, index=False)
    print(f"All files appended successfully into {output_file}")


def add_components(version, components):
    # Read the CSV file
    df = pd.read_csv("../../data/ReleaseNotes/Isis/release_notes_isis_merged.csv")

    try:
        # Ensure that the version exists
        if version in df["version"].values:
            # Assign the list directly to the 'component' column where version matches
            df.loc[df["version"] == version, "component"] = [components]
        else:
            print(f"Version {version} not found in the 'version' column")
    except KeyError:
        print("Error: 'version' column not found in the DataFrame")

    df.to_csv("../data/ReleaseNotes/Isis/release_notes_isis_merged.csv", index=False)
    print("Components added successfully")


def main():
    pass
    # Merge isis release notes with newer release notes
    input_folder = "../data/ReleaseNotes/Isis"  # Replace with the folder path where your CSV files are located
    input_files_pattern = "release_notes_isis*.csv"  # This matches all CSV files with the specified pattern
    output_file = "../../data/ReleaseNotes/Isis/release_notes_isis_merged.csv"  # Output file where the merged data will be saved

    append_csv_files(input_folder, input_files_pattern, output_file)
    # Call the function with the URL and name
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2018/2.0.0-M1/relnotes.html", "isis_2.0.0-M1")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2019/2.0.0-M2/relnotes.html", "isis_2.0.0-M2")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2020/2.0.0-M3/relnotes.html", "isis_2.0.0-M3")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2020/2.0.0-M4/relnotes.html", "isis_2.0.0-M4")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2021/2.0.0-M5/relnotes.html", "isis_2.0.0-M5")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2021/2.0.0-M6/relnotes.html", "isis_2.0.0-M6")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2022/2.0.0-M7/relnotes.html", "isis_2.0.0-M7")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2022/2.0.0-M8/relnotes.html", "isis_2.0.0-M8")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2022/2.0.0-M9/relnotes.html", "isis_2.0.0-M9")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2023/2.0.0-RC1/relnotes.html", "isis_2.0.0-RC1")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2023/2.0.0-RC2/relnotes.html", "isis_2.0.0-RC2")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2023/2.0.0-RC3/relnotes.html", "isis_2.0.0-RC3")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2024/2.0.0-RC4/relnotes.html", "isis_2.0.0-RC4")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2024/2.0.0/relnotes.html", "isis_2.0.0")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2024/3.0.0/relnotes.html", "isis_3.0.0")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2024/2.1.0/relnotes.html", "isis_2.1.0")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2024/3.1.0/relnotes.html", "isis_3.1.0")
    # get_release_notes("https://causeway.apache.org/relnotes/latest/2025/3.2.0/relnotes.html", "isis_3.2.0")

    add_components("1.16.1", ["1.16.11"])
    add_components("1.16.0", ["1.16.0"])
    add_components("1.15.1", ["1.15.1"])
    add_components("1.15.0", ["1.15.0"])
    add_components("1.14.0", ["1.14.0"])
    add_components("1.13.2.1", ["1.13.2.1"])
    add_components("1.13.2", ["1.13.2"])
    add_components("1.13.1", ["1.13.1"])
    add_components("1.13.0", ["1.13.0"])
    add_components("1.12.2", ["1.12.2"])
    add_components("1.12.1", ["1.12.1"])
    add_components("1.12.0", ["1.12.0"])
    add_components("2.0.0-M1", ["2.0.0-M1"])
    add_components("2.0.0-M2", ["2.0.0-M2"])
    add_components("2.0.0-M3", ["2.0.0-M3"])
    add_components("2.0.0-M4", ["2.0.0-M4"])
    add_components("2.0.0-M5", ["2.0.0-M5"])
    add_components("2.0.0-M6", ["2.0.0-M6"])
    add_components("2.0.0-M7", ["2.0.0-M7"])
    add_components("2.0.0-M8", ["2.0.0-M8"])
    add_components("2.0.0-M9", ["2.0.0-M9"])
    add_components("2.0.0-RC1", ["2.0.0-RC1"])
    add_components("2.0.0-RC2", ["2.0.0-RC2"])
    add_components("2.0.0-RC3", ["2.0.0-RC3"])
    add_components("2.0.0-RC4", ["2.0.0-RC4"])
    add_components("2.0.0", ["2.0.0"])
    add_components("3.0.0", ["3.0.0"])
    add_components("2.1.0", ["2.1.0"])
    add_components("3.1.0", ["3.1.0"])
    add_components("3.2.0", ["3.2.0"])
    add_components("1.11.1", ["1.11.1"])
    add_components("1.11.0", ["1.11.0"])
    add_components("1.10.0", ["1.10.0"])
    add_components("1.9.0", ["1.9.0"])
    add_components("1.8.0", ["1.8.0", "core-1.8.0", "archetype-1.8.0"])
    add_components("1.7.0",
                   ["1.7.0", "core-1.7.0", "viewer-wicket-1.7.0", "archetype-simpleapp-1.7.0", "archetype-todoapp-1.7.0"])
    add_components("1.6.0",
                   ["1.6.0", "core-1.6.0", "viewer-wicket-1.6.0", "archetype-simpleapp-1.6.0", "archetype-todoapp-1.6.0"])
    add_components("1.5.0",
                   ["1.5.0", "core-1.5.0", "objectstore-jdo-1.5.0", "security-shiro-1.5.0", "viewer-restfulobjects-2.3.0",
                    "viewer-wicket-1.5.0", "archetype-simple-wrj-1.5.0", "archetype-quickstart-wrj-1.5.0"])
    add_components("1.4.1", ["1.4.1", "objectstore-jdo-1.4.1", "viewer-wicket-1.4.1", "archetype-simple-wrj-1.4.1",
                             "archetype-quickstart-wrj-1.4.1"])
    add_components("1.4.0", ["1.4.0", "core-1.4.0", "objectstore-jdo-1.4.0", "security-file-1.4.0", "security-shiro-1.4.0",
                             "viewer-restfulobjects-2.2.0", "viewer-wicket-1.4.0", "archetype-simple-wrj-1.4.0",
                             "archetype-quickstart-wrj-1.4.0"])
    add_components("1.3.1", ["1.3.1", "viewer-wicket-1.3.1", "archetype-simple-wrj-1.3.1", "archetype-quickstart-wrj-1.3.1"])
    add_components("1.3.0", ["1.3.0", "core-1.3.0", "objectstore-jdo-1.3.0", "security-file-1.0.2", "security-shiro-1.3.0",
                             "viewer-restfulobjects-2.1.0", "viewer-wicket-1.3.0", "archetype-simple-wrj-1.3.0",
                             "archetype-quickstart-wrj-1.3.0"])
    add_components("1.2.0", ["1.2.0", "core-1.2.0", "objectstore-jdo-1.2.0", "security-file-1.0.1", "security-shiro-1.1.1",
                             "viewer-restfulobjects-2.0.0", "viewer-wicket-1.2.0", "archetype-wrj-1.0.3"])
    add_components("1.1.0", ["1.1.0", "core-1.1.0", "security-shiro-1.1.0", "viewer-wicket-1.1.0", "archetype-wrj-1.0.2"])
    add_components("1.0.1", ["1.0.1", "security-shiro-1.0.0", "archetype-wrj-1.0.1"])
    add_components("1.0.0", ["1.0.0", "core-1.0.0", "security-file-1.0.0", "viewer-wicket-1.0.0", "viewer-restfulobjects-1.0.0",
                             "archetype-wrj-1.0.0"])


if __name__ == "__main__":
    main()
