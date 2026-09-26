import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis (EDA) - Business Entity Resolution\n",
    "This notebook contains a basic EDA to understand the dataset for the Business Entity Resolution challenge."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# File paths\n",
    "s1_path = '../dataset/train/train_source1.tsv'\n",
    "s2_path = '../dataset/train/train_source2.tsv'\n",
    "s3_path = '../dataset/train/train_source3.tsv'\n",
    "gt_path = '../dataset/train/train_ground_truth.tsv'\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Data\n",
    "Since the datasets are large (~5M rows for S2/S3), we'll read a sample for initial exploration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load a sample of the data to avoid memory issues during initial EDA\n",
    "sample_size = 100000\n",
    "df_s1 = pd.read_csv(s1_path, sep='\\t', nrows=sample_size)\n",
    "df_s2 = pd.read_csv(s2_path, sep='\\t', nrows=sample_size)\n",
    "df_s3 = pd.read_csv(s3_path, sep='\\t', nrows=sample_size)\n",
    "df_gt = pd.read_csv(gt_path, sep='\\t', nrows=sample_size)\n",
    "\n",
    "print(f\"Source 1 Sample Size: {len(df_s1)}\")\n",
    "print(f\"Source 2 Sample Size: {len(df_s2)}\")\n",
    "print(f\"Source 3 Sample Size: {len(df_s3)}\")\n",
    "display(df_s1.head())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Missing Values\n",
    "Check for missing business names and addresses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Missing values in Source 1:\")\n",
    "print(df_s1.isnull().sum())\n",
    "\n",
    "print(\"\\nMissing values in Source 2:\")\n",
    "print(df_s2.isnull().sum())\n",
    "\n",
    "print(\"\\nMissing values in Source 3:\")\n",
    "print(df_s3.isnull().sum())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Country Distribution\n",
    "Let's check the distribution of countries across the sources."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "\n",
    "df_s1['country'].value_counts().plot(kind='bar', ax=axes[0], title='Source 1 Countries')\n",
    "df_s2['country'].value_counts().plot(kind='bar', ax=axes[1], title='Source 2 Countries')\n",
    "df_s3['country'].value_counts().plot(kind='bar', ax=axes[2], title='Source 3 Countries')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Ground Truth Analysis (Singletons & Cardinality)\n",
    "Analyze how many matches each S1 entity has."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill NaN with empty string for matches\n",
    "df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].fillna('')\n",
    "\n",
    "# Count number of matches (comma separated)\n",
    "df_gt['match_count'] = df_gt['matched_entity_ids'].apply(lambda x: len(str(x).split(',')) if x != '' else 0)\n",
    "\n",
    "print(\"Match Count Summary:\")\n",
    "print(df_gt['match_count'].describe())\n",
    "\n",
    "singletons = (df_gt['match_count'] == 0).sum()\n",
    "print(f\"\\nNumber of Singletons (0 matches) in sample: {singletons} ({(singletons/len(df_gt))*100:.2f}%)\")\n",
    "\n",
    "df_gt['match_count'].value_counts().sort_index().plot(kind='bar', figsize=(10, 5), title='Distribution of Match Counts')\n",
    "plt.xlabel('Number of Matches')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Token Analysis & Address Variations\n",
    "Analyze common words in names and addresses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import Counter\n",
    "import re\n",
    "\n",
    "def get_top_tokens(series, n=20):\n",
    "    words = series.dropna().str.lower().apply(lambda x: re.findall(r'\\b\\w+\\b', x)).sum()\n",
    "    return Counter(words).most_common(n)\n",
    "\n",
    "# This can be slow, so we just sample a small subset\n",
    "print(\"Top Business Name tokens in Source 1 (sample):\")\n",
    "print(get_top_tokens(df_s1['business_name'].head(5000)))\n",
    "\n",
    "print(\"\\nTop Address tokens in Source 1 (sample):\")\n",
    "print(get_top_tokens(df_s1['business_address'].head(5000)))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('/home/rishang/MLC26/EDA.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated.")
