import pandas as pd
df = pd.read_csv('C:/Users/shagu/Desktop/pythonProject2/solardata1.csv')
# identify missing values
df.isnull().sum()
# Fill or Drop Missing Values: as i cannot drop values bz data set is not that big
df.fillna("N/A" , inplace=True)
#Standardizing Text:
# Convert text to lower case:
df["Comment/Review Text"] = df["Comment/Review Text"].str.lower()
# Remove unnecessary spaces:
df["Comment/Review Text"] = df["Comment/Review Text"].str.strip()
"""print(df.head())
 print(df.to_string())
 print(df.info())"""
# Handling Duplicates:
# Identify Duplicates
duplicates = df[df.duplicated()] # no duplicates was found , but anyways remove duplicates
df.drop_duplicates(inplace=True)
# Categorical Data Standardization:
# Standardize Ratings (e.g., converting '1-star' to '1'):
df['Rating'] = df['Rating'].replace({
    '1-star': 1,
    '5-star': 5,
    # add other mappings if necessary
})
# Export Cleaned Data:
df.to_csv('cleaned_solardata1.csv', index=False)

# .........................................................................................
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Load the dataset
df = pd.read_csv('solardata1.csv')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Tokenization - break the sentences into words or tokens
df['Tokens'] = df['Comment/Review Text'].apply(lambda x: word_tokenize(x.lower()))

# Remove stop words eg: the , is etc
df['Tokens'] = df['Tokens'].apply(lambda tokens: [word for word in tokens if word.isalnum() and word not in stop_words])

# Stemming - eg running = run
df['Tokens'] = df['Tokens'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# Print the first few rows to check
print(df.head())
# print(df.to_string())
# Optionally, save the processed data to a new CSV file
df.to_csv('processed_solar_data1.csv', index=False)

# .........................................................................................................
# Exploratory Data Analysis (EDA)
# Descriptive Statistics: Get a summary of the data.
# Distribution Analysis: Analyze the distribution of sentiments.
# Visualization: Use plots to visualize the data.
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe(include='all'))
# 2. Distribution of Source
source_counts = df['Source'].value_counts()
plt.figure(figsize=(10, 6))
# Create a barplot
sns.barplot(x=source_counts.index, y=source_counts.values, palette='viridis', hue=None, legend=False)
# palette: This argument controls the color scheme of the plot. It determines how different categories or values are represented using colors.
# 'viridis': This is one of the built-in color palettes provided by Seaborn. 'viridis' is a color map that ranges
# from yellow to purple, designed to be perceptually uniform. It’s often used for continuous data.
plt.title('Distribution of Reviews by Source')
plt.xlabel('Source')
plt.ylabel('Count')
plt.show()
# 3. Sentiment Distribution Analysis (If applicable)
if 'Sentiment' in df.columns:
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', hue=None, legend=False)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
else:
    print("Sentiment column not found in the dataset.")
# 4. Most Common Words in the Tokenized Text
all_tokens = [word for tokens in df['Tokens'] for word in tokens]
token_freq = nltk.FreqDist(all_tokens)
# Plotting the 30 most common words
plt.figure(figsize=(12, 8))  #figsize=(12, 8) creates a plot that is 12 inches wide and 8 inches tall
token_freq.plot(30, cumulative=False)
plt.title('Top 30 Most Common Words in Reviews/Comments')
plt.show()

df.to_csv('processed_solar_data.csv', index=False)