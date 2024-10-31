# GitHub Users in Chennai

This repository contains data about GitHub users in Chennai with over 50 followers and their repositories.

**GitHub Data Scraper**
This project leverages GitHub’s API to retrieve and analyze repository and user data, offering insights into developer trends and engagement.
The gitscrap.py script performs data scraping tasks, focusing on user activity and repository statistics to uncover meaningful patterns.
Insights from this analysis highlight key areas for optimizing developer workflows and enhancing project visibility.
Project Overview

1. Data Scraping Process
The gitscrap.py script connects to the GitHub API, retrieving comprehensive datasets on both repositories and users. Utilizing Python’s requests library, the script iteratively fetches data by navigating through pagination to ensure the capture of all relevant records. Each data point is parsed and organized into a structured format before saving it as a CSV for analysis.
- Data collected using GitHub API
- Date of collection: 2024-10-30
- Only included users with 50+ followers
- Up to 500 most recently pushed repositories per user

2. Key Insights from Data Analysis
Upon analyzing the scraped data, one of the most surprising findings was the correlation between the number of forks and repository longevity. Repositories with higher fork counts tend to show consistent activity and maintenance, which often correlates with longer-term project relevance.

3. Recommendations for Developers
Developers can increase their project’s visibility by focusing on documentation quality and encouraging contributions from newcomers. Open issues and welcoming contribution guidelines significantly increase engagement rates, leading to a more robust project lifecycle.

## Files

1. `users.csv`: Contains information about 433 GitHub users in Chennai with over 50 followers
2. `repositories.csv`: Contains information about 26947 public repositories from these users
3. `gitscrap.py`: Python script used to collect this data


