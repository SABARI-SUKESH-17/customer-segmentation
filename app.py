
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Dashboard")

df = pd.read_csv("customer_data.csv")

st.subheader("Customer Data")
st.dataframe(df)

X = df[['AnnualIncome', 'SpendingScore']]

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

st.subheader("Customer Segments")

fig, ax = plt.subplots()

for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'], label=f'Cluster {cluster}')

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.legend()

st.pyplot(fig)

st.subheader("Cluster Summary")
st.dataframe(df.groupby('Cluster')[['AnnualIncome', 'SpendingScore']].mean())
