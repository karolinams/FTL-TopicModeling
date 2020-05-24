#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#Dataset upload
data_karolina = pd.read_csv("corona.csv", encoding="utf-8",error_bad_lines=False, names=["count", "Date", "location", "Tweet"])
data_karolina = data_karolina.drop(columns=['count'], axis=1)
data_karolina
#data_karolina.to_csv("final.csv", sep=";", na_rep="", index=False, encoding="utf-8-sig")


# In[2]:


# Load the regular expression library
import re
# Remove punctuation
data_karolina['Tweet'] = data_karolina['Tweet'].map(lambda x: re.sub('[,\.!?@]', '', x))
# Convert the titles to lowercase
data_karolina['Tweet'] = data_karolina['Tweet'].map(lambda x: x.lower())
# Print out the first rows 
data_karolina['Tweet'].head()


# In[4]:


# Import the wordcloud library
import matplotlib
import wordcloud
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(data_karolina['Tweet'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
vis = wordcloud.to_image()
vis.show()
