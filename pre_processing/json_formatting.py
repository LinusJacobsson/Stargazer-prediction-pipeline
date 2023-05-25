# Code snippet to format the previous gathered
# JSON file into a single Pandas dataframe.
# Usage: python3 json_formatting.py filename (which is merged_repos.json here)
# This will return and print the final dataframe.
# Written by Linus Jacobsson 19/5 2023


import json
import sys
import pandas as pd

filename = sys.argv[1]

# Load the data from the file
with open(filename, 'r') as f:
    data = json.load(f)

# Extract the repository data
repos = data['data']['search']['edges']

# Initialize a list to hold dictionaries
repo_dicts = []

# Iterate over repos and extract relevant info
for i, repo in enumerate(repos, start=1):
    repo_data = repo['node']

    # Create a dictionary for each repository
    repo_dict = {}

    # Basic info
    repo_dict['Repository Name'] = repo_data['name']
    repo_dict['Owner'] = repo_data['owner']['login']
    repo_dict['Star Count'] = repo_data['stargazers']['totalCount']

    # Additional features
    repo_dict['Fork Count'] = repo_data['forkCount']
    repo_dict['Created at'] = repo_data['createdAt']
    repo_dict['Updated at'] = repo_data['updatedAt']
    repo_dict['Primary Language'] = repo_data['primaryLanguage']['name'] if repo_data['primaryLanguage'] else 'None'
    repo_dict['PR Count'] = repo_data['pullRequests']['totalCount']
    repo_dict['Issue Count'] = repo_data['issues']['totalCount']
    repo_dict['Watcher Count'] = repo_data['watchers']['totalCount']
    repo_dict['Disk Usage'] = repo_data['diskUsage']
    repo_dict['Is Fork'] = repo_data['isFork']
    repo_dict['Is Archived'] = repo_data['isArchived']
    repo_dict['License Info'] = repo_data['licenseInfo']['name'] if repo_data['licenseInfo'] else 'None'

    # Extract the topics
    repo_dict['Topics'] = ', '.join(edge['node']['topic']['name'] for edge in repo_data['repositoryTopics']['edges'])

    # Add the dictionary to the list
    repo_dicts.append(repo_dict)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(repo_dicts)
# Save to CSV file
df.to_csv('data.csv')
