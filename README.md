# Disease Clustering using Ayurvedic Data

Unsupervised machine learning pipeline to discover hidden clusters of diseases based on symptoms, lifestyle factors, and environmental influences.

## Problem Statement

Healthcare datasets often hide meaningful relationships between diseases and health attributes. This project applies **unsupervised clustering** to group diseases by shared symptom and lifestyle patterns — without using any predefined labels.

The solution was evaluated using the **Silhouette Score** (higher = better-defined clusters).

## Dataset Features

Each record represents a disease profile described by:

- **Symptom patterns**
- **Lifestyle factors**: Sleep habits, stress levels, physical activity, dietary influences
- **Environmental influences**
- **Treatment-related details**

## Approach

### 1. Feature Engineering
- Constructed derived features capturing relationships between symptom severity and lifestyle patterns
- Encoded categorical medical attributes for clustering compatibility

### 2. Feature Selection
- Applied mutual information and correlation filtering
- Used PCA / dimensionality reduction to reduce noise and improve cluster separability

### 3. Clustering
- Experimented with K-Means, DBSCAN, and Agglomerative Clustering
- Heuristic optimization of number of clusters to maximize Silhouette Score
- Final clusters represent diseases with similar symptom-lifestyle co-occurrence patterns

### 4. Visualization & Interpretation
- Visualized clusters using t-SNE and PCA plots
- Interpreted cluster characteristics to surface actionable health patterns

## Tech Stack

- **Language**: Python
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn

## Submission Format

```
Disease,cluster
Cough,0
Diabetes,1
Hypertension,1
Migraine,2
```

## Results

Achieved well-separated clusters with a strong Silhouette Score through iterative feature engineering and heuristic cluster count tuning.
