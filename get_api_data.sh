#!/bin/bash

# Script that request the information of the top 1000 most popular Github 
# repositories, based on number of stargazers.

# Usage: Simply run the script without any arguments. Note! This assumes that
# you are already authenticated for Github on the working machine. It will return
# a single file called top_repos.json.

# Written by Linus Jacobsson 18/5 2023

query='
{
  search(query: "stars:>1", type: REPOSITORY, first: 100) {
    edges {
      node {
        ... on Repository {
          name
          owner {
            login
          }
          stargazers {
            totalCount
          }
          forkCount
          createdAt
          updatedAt
          primaryLanguage {
            name
          }
          pullRequests {
            totalCount
          }
          issues {
            totalCount
          }
          watchers {
            totalCount
          }
          diskUsage
          isFork
          isArchived
          licenseInfo {
            name
          }
          repositoryTopics(first: 5) {
            totalCount
            edges {
              node {
                topic {
                  name
                }
              }
            }
          }
        }
      }
    }
    pageInfo {
      endCursor
    }
  }
}'

# We want to make the inital request, in order to get the cursor.
gh api graphql -f query="$query" > top_repos_1.json

# Now we can extract the cursor pointing to the next page of results.
cursor=$(jq -r '.data.search.pageInfo.endCursor' < top_repos_1.json)

# We'll now loop through 9 times to get the remaining 900 entries
for i in {2..10}; do
  sleep 5 # We'll wait a few seconds to avoid getting timed out.
  # Now we can use the retrieved cursor to modify our query.
  query='
{
  search(query: "stars:>1", type: REPOSITORY, first: 100, after: "'$cursor'") {
    edges {
      node {
        ... on Repository {
          name
          owner {
            login
          }
          stargazers {
            totalCount
          }
          forkCount
          createdAt
          updatedAt
          primaryLanguage {
            name
          }
          pullRequests {
            totalCount
          }
          issues {
            totalCount
          }
          watchers {
            totalCount
          }
          diskUsage
          isFork
          isArchived
          licenseInfo {
            name
          }
          repositoryTopics(first: 10) {
            totalCount
            edges {
              node {
                topic {
                  name
                }
              }
            }
          }
        }
      }

    }
    pageInfo {
      endCursor
    }
  }
}'


  # Now we can make the next query and store the result in a new file.
  gh api graphql -f query="$query" > "top_repos_$i.json"

  # We will also need to get the next cursor
  cursor=$(jq -r '.data.search.pageInfo.endCursor' < "top_repos_$i.json")
  done

# Now we will have the different 10 files, each with 100 results. Now we
# want to gather them to a single file.

# Inspired by: https://stackoverflow.com/questions/29636331/merging-json-files-using-a-bash-script
jq -s 'reduce .[] as $item ({"data": {"search": {"edges": []}}}; .data.search.edges += $item.data.search.edges)' top_repos_*.json > merged_repos.json
