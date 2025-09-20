
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

# Load the metadata
df = pd.read_csv('metadata.csv')

# Basic exploration
print("Dataset shape:", df.shape)
print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.describe())

# -------------------------
# Part 2: Data Cleaning and Preparation
# -------------------------
# Convert date column to datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Extract year from publish_time
df['year'] = df['publish_time'].dt.year

# Example: create abstract word count
df['abstract_word_count'] = df['abstract'].fillna('').apply(lambda x: len(str(x).split()))

# Drop columns with too many missing values (optional)
# df_clean = df.drop(columns=['some_column_with_many_nans'])
df_clean = df.copy()

# -------------------------
# Part 3: Data Analysis and Visualization
# -------------------------
# Publications per year
year_counts = df_clean['year'].value_counts().sort_index()
plt.figure(figsize=(8,5))
plt.bar(year_counts.index, year_counts.values, color='skyblue', edgecolor='black')
plt.title('Publications by Year')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.show()

# Top journals
top_journals = df_clean['journal'].value_counts().head(10)
plt.figure(figsize=(10,6))
top_journals.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Top 10 Journals Publishing COVID-19 Research')
plt.xlabel('Journal')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45)
plt.show()

# Word cloud for paper titles
text = ' '.join(df_clean['title'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Distribution by source
source_counts = df_clean['source_x'].value_counts()
plt.figure(figsize=(6,4))
source_counts.plot(kind='bar', color='orange')
plt.title('Paper Counts by Source')
plt.show()

# -------------------------
# Part 4: Streamlit Application
# -------------------------
# To run: save this as streamlit_app.py and run `streamlit run streamlit_app.py`

st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

# Interactive year selection
min_year = int(df_clean['year'].min())
max_year = int(df_clean['year'].max())
year_range = st.slider("Select publication year range", min_year, max_year, (2020, 2021))

# Filter data
filtered_df = df_clean[(df_clean['year'] >= year_range[0]) & (df_clean['year'] <= year_range[1])]

st.write(f"Displaying papers from {year_range[0]} to {year_range[1]}")
st.dataframe(filtered_df.head())

# Plot publications over time
st.write("Publications by Year")
year_counts_filtered = filtered_df['year'].value_counts().sort_index()
st.bar_chart(year_counts_filtered)

# Top journals
st.write("Top Journals")
top_journals_filtered = filtered_df['journal'].value_counts().head(10)
st.bar_chart(top_journals_filtered)

# Word cloud in Streamlit
st.write("Word Cloud of Paper Titles")
st.image(wordcloud.to_array())
