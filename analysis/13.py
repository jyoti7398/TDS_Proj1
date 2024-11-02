import pandas as pd
import statsmodels.api as sm
import os
import sys
from io import StringIO

# Ensure UTF-8 encoding for output
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load the CSV file with UTF-8 encoding
csv_file = 'users.csv'  # Ensure this path is correct
df = pd.read_csv(csv_file, encoding='utf-8')

# Use StringIO to capture the DataFrame output with utf-8 encoding
output = StringIO()
df.head().to_string(buf=output)  # Save the DataFrame preview to the buffer

# Print the buffer's content as utf-8 to handle special characters
print("DataFrame Overview:")
sys.stdout.buffer.write(output.getvalue().encode('utf-8'))
print("\nDataFrame Info:")
print(df.info())

# Filter out users without bios
df = df[df['bio'].notnull()]

# Calculate the length of each bio in words
df['bio_word_count'] = df['bio'].str.split().str.len()

# Prepare the independent variable (X) and dependent variable (y)
X = df['bio_word_count']
y = df['followers']  # Adjust the column name as per your dataset

# Add a constant to the independent variable (for the intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the slope (coefficient of the bio_word_count)
slope = model.params['bio_word_count']

# Print the regression slope rounded to three decimal places
print(f"\nRegression slope of followers on bio word count: {slope:.3f}")
