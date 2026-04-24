import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# Lets start by dropping unnecessary columns
df = df.drop(['Hindi Name', 'Marathi Name'], axis=1)
# print(df)
# Now lets drop a few more columns by knowing their exact reasons

# Know this that if we are making clusters and have the disease into that do you think if the 
    #medication is same then should be they in the same cluster. So we'll remove the features such as
    #Ayurvedic Herbs, Yoga & Physical Therapy, Diet and Lifestyle Recommendations and Occupation and Lifestyle

df = df.drop(['Ayurvedic Herbs', 'Yoga & Physical Therapy', 'Diet and Lifestyle Recommendations', 'Occupation and Lifestyle'], axis=1)
# print(df)

'''
Diagnosis & Tests, Duration of Treatment — too clinical/procedural
Cultural Preferences — too subjective
Herbal/Alternative Remedies — treatment
Formulation
'''

df = df.drop(['Diagnosis & Tests', 'Duration of Treatment', 'Cultural Preferences', 'Herbal/Alternative Remedies', 'Formulation'], axis=1)
# print(df)

df = df.drop(['Current Medications', 'Medical Intervention', 'Prevention', 'Patient Recommendations', 'Prognosis'], axis=1)

'''
print(df.columns.tolist())
print("\n")
for col in df.columns:
    print(f"{col}: {df[col].unique()[:3]}")
'''

# We could witness few upper and lowercase stuff but is exact same word so to solve that:
# Know each thing before that
    # applymap(): goes to each cell in the entire df    (applymap is now old lets replace it by map)
    # lambda: is an annonymous function for maybe whatever we wish to do with the data we'll have it as x
    # x.strip: removes spaces from start and end
    # isinstance: checks whether a value belongs to a particular data type
df = df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)

'''
Now looking at what remains, here's how I'd categorize your columns:
Ordinal — Symptom Severity, Stress Levels, Physical Activity Levels
Multi-value — Symptoms, Medical History, Risk Factors, Environmental Factors, Family History, Dietary Habits, Allergies, Seasonal Variation, Doshas, Complications
Single categorical — Sleep Patterns, Age Group, Gender, Constitution/Prakriti
'''

# Now we'll make all the texts numerical

# Let's start with the easiest:     Ordinal encoding
'''
Look at these three columns:

Symptom Severity: Mild < Moderate < Severe
Stress Levels: Low < Moderate < High
Physical Activity Levels: Low < Moderate < High

But still i suspect that there could be some extra values like "low to moderate" which might not come under
a fixed ordinality, So'''

'''# Used to know what is the variety of the entries in a column(by name) but if you want count use nunique instead of unique
print(df['Symptom Severity'].unique())
print(df['Stress Levels'].unique())
print(df['Physical Activity Levels'].unique())
# We were right we got more than these required values, So we will have respective mapping'''


Severity_map = {
    'mild': 1,
    'mild to moderate': 1.5,
    'moderate': 2,
    'moderate to high': 2.5,
    'high': 3,
    'moderate to severe': 2.5,
    'mild to severe': 2,
    'severe': 3
}

Stress_map = {
    'low stress': 1,
    'moderate stress': 2,
    'high stress': 3,
    'very high stress': 3.5
}

Activity_map = {
    'low': 1,
    'low to moderate': 1.5,
    'moderate to low': 1.5,
    'moderate': 2,
    'moderate to high': 2.5,
    'high': 3
}

sleep_map = {
    'regular sleep': 1,
    'irregular sleep': 2,
    'disrupted sleep': 3,
    'poor sleep': 4,
    'poor sleep quality': 4,
    'extreme fatigue': 5
}



df['Symptom Severity'] = df['Symptom Severity'].map(Severity_map)
df['Stress Levels'] = df['Stress Levels'].map(Stress_map)
df['Physical Activity Levels'] = df['Physical Activity Levels'].map(Activity_map)
df['Sleep Patterns'] = df['Sleep Patterns'].map(sleep_map)


'''
print(df[['Symptom Severity', 'Stress Levels', 'Physical Activity Levels']].head())
print("\nAny nulls?")
print(df[['Symptom Severity', 'Stress Levels', 'Physical Activity Levels']].isnull().sum())
print(df['Sleep Patterns'].unique())
print("Nulls:", df['Sleep Patterns'].isnull().sum())


for col in ['Age Group', 'Gender', 'Constitution/Prakriti']:
    print(f"\n{col}:")
    print(df[col].unique())'''


# Now we'll do a case of one-hot encoding

# Gender
gender_male_map = {
    'male': 1,
    'males': 1,
    'female': 0,
    'mostly male': 0.75,
    'mostly female': 0.25,
    'all genders': 0.5,
    'both genders': 0.5
}

gender_female_map = {
    'male': 0,
    'males': 0,
    'female': 1,
    'mostly male': 0.25,
    'mostly female': 0.75,
    'all genders': 0.5,
    'both genders': 0.5
}

df['Gender_Male'] = df['Gender'].map(gender_male_map)
df['Gender_Female'] = df['Gender'].map(gender_female_map)
df = df.drop('Gender', axis=1)


'''# verify
print(df[['Gender_Male', 'Gender_Female']].nunique()) #witnessed an error when used unique() bcoz it doesn't works with multiple columns
print("Nulls:", df[['Gender_Male', 'Gender_Female']].isnull().sum())'''





'''
Now we'll do the most difficult(according to me) fixing in the column named "Age Group", which has 
around 54 distint entries for which we'll use some another library called "re" (regex). It is used
pattern recognition'''

import re

def extract_age_range(value):                       # Making a function for that
    numbers = re.findall(r'\d+', value)             # finds all values with pattern 
    '''\d means to find a number(0-9) from the whole thing(value in here), + is a quantifier in regex which 
    check more than one occurance of a number'''
    
    if len(numbers) == 2:                           # if the list has 2 values then it'll make it lower and upper value of the range
        # two numbers found
        return int(numbers[0]), int(numbers[1])     # would return a tuple
    
    elif len(numbers) == 1:
        # one number found, likely '40+ years'
        return (int(numbers[0]), 90)
    
    else:
        # no numbers, use keyword matching; regex is not being used here
        if 'infant' in value:
            return 0, 2
        elif 'child' in value:
            return (2,12)
        elif 'teen' in value:
            return (12, 18)
        elif 'adult' in value:
            return (18, 45)
        elif 'elderly' in value:
            return (45, 90)
        else:
            # all ages, any age, all age groups
            return (0, 90)
        
# Now we'll feed the values in our two new columns namely age_min and age_max
df['age_min'] = df['Age Group'].apply(lambda x: extract_age_range(x)[0])
df['age_max'] = df['Age Group'].apply(lambda x: extract_age_range(x)[1])

# Lets remove the Age Group column now as we don't need it
df = df.drop('Age Group', axis=1)


'''Verification
print(df[['age_min', 'age_max']].head(10))
print("Nulls:", df[['age_min', 'age_max']].isnull().sum())

# check what happened to tricky values
print(extract_age_range('adults, elderly'))
print(extract_age_range('adults (females)'))
print(extract_age_range('teenagers to 30s'))

print(df['Constitution/Prakriti'].unique())'''


for dosha in ['vata', 'pitta', 'kapha']:
    df[f'dosha_{dosha}'] = df['Doshas'].apply(
        lambda x: 1 if dosha in x else 0
    )

df = df.drop(['Doshas', 'Constitution/Prakriti'], axis=1)

'''# verify
print(df[['dosha_vata', 'dosha_pitta', 'dosha_kapha']].head())


print(df['Symptoms'].unique()[:5])
print(df['Allergies (Food/Env)'].unique()[:5])'''




# Now you know about one-hot enconding but in the column Symptoms we have a large number of values
# for that we will now use a Scikit learn library named as MultiLabelBinarizer
# here it will be like what we did with the gender thing but instead of different columns it stores 1 0 1 1 0.. in the same column

from sklearn.preprocessing import MultiLabelBinarizer

# first fill nan in allergies specifically
df['Allergies (Food/Env)'] = df['Allergies (Food/Env)'].fillna('no allergies')

mlb = MultiLabelBinarizer()

multi_cols = ['Symptoms', 'Medical History', 'Risk Factors', 
              'Environmental Factors', 'Family History', 'Dietary Habits',
              'Allergies (Food/Env)', 'Seasonal Variation', 'Complications']

for col in multi_cols:
    # fill remaining nans with unknown for other columns
    df[col] = df[col].fillna('unknown')
    
    # split each cell by comma into a list
    split_col = df[col].apply(lambda x: [i.strip() for i in x.split(',')])
    
    # apply mlb
    encoded = mlb.fit_transform(split_col)
    
    # create new columns with prefix
    encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{c}" for c in mlb.classes_])
    
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    df = df.drop(col, axis=1)

# print(df.shape)
# print("Nulls:", df.isnull().sum().sum())          Here we got 2060 columns and that is way too much so now we'll use PCA




# see which original column exploded into most columns
for col in ['Symptoms', 'Medical History', 'Risk Factors', 
              'Environmental Factors', 'Family History', 'Dietary Habits',
              'Allergies (Food/Env)', 'Seasonal Variation', 'Complications']:
    count = len([c for c in df.columns if c.startswith(col)])
    #print(f"{col}: {count} columns")        # We got 462 columns. These are more that the number of rows so

# if a binary column has less than X% of rows as 1
# then it's too rare to be meaningful → drop it

# for all binary columns, what percentage of rows are 1?
binary_cols = [c for c in df.columns if df[c].nunique() == 2]
coverage = df[binary_cols].mean() * 100


# print(f"Total binary columns: {len(binary_cols)}")
# print(f"\nCoverage distribution:")
# print(coverage.describe())

'''This is very revealing! Look at the numbers carefully:

Mean coverage is only 0.89% — meaning on average each binary column is 1 for less than 1% of diseases
75th percentile is 0.44% — meaning 75% of all binary columns appear in less than 2 diseases out of 446!

That's enormous noise. Most of these columns are essentially useless for clustering.
Now here's the judgment call for you — what threshold should we use to keep a column?
Think about it this way:

If a symptom appears in only 1 disease out of 446, does it help clustering? No
If a symptom appears in 50 diseases, it's a meaningful pattern worth keeping

So the question is — what's the minimum percentage of diseases a feature should appear in to be worth keeping?
'''


for threshold in [1, 2, 3, 5, 10]:
    kept = (coverage >= threshold).sum()
    # print(f"Threshold {threshold}%: keeps {kept} columns")

non_binary_cols = [c for c in df.columns if df[c].nunique() != 2]
# print(f"Non binary columns: {len(non_binary_cols)}")
# print(non_binary_cols)


for threshold in [3, 5]:
    kept_binary = (coverage >= threshold).sum()
    total = kept_binary + len(non_binary_cols)
    # print(f"Threshold {threshold}%: {kept_binary} binary + {len(non_binary_cols)} non-binary = {total} total columns")


# separate Disease column first
disease_col = df['Disease'].copy()

# keep only columns above 5% threshold
cols_to_keep = list(coverage[coverage >= 5].index)
cols_to_keep = cols_to_keep + non_binary_cols
cols_to_keep.remove('Disease')

df_filtered = df[cols_to_keep]

# print(f"Final shape: {df_filtered.shape}")
# print(f"Columns: {df_filtered.columns.tolist()}")

# We could see same things with a bit of difference so lets solve that before PCA
df_filtered = df_filtered.rename(columns={
    'Risk Factors_genetic mutations': 'Risk Factors_genetic mutation',
    'Risk Factors_genetics': 'Risk Factors_genetic mutation',
    'Dietary Habits_balanced diet': 'Dietary Habits_balanced',
    'Dietary Habits_high-fiber diet': 'Dietary Habits_high-fiber',
    'Dietary Habits_high-protein diet': 'Dietary Habits_high-protein',
})

# merge duplicate columns by taking max value
df_filtered = df_filtered.T.groupby(level=0).max().T
# print(df_filtered.shape)





# Now lets start pca
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtered)

# print(f"Scaled shape: {df_scaled.shape}")
# print(f"Mean of first column: {df_scaled[:, 0].mean():.4f}")  
# print(f"Std of first column: {df_scaled[:, 0].std():.4f}")


pca = PCA()
pca.fit(df_scaled)

# cumulative variance explained
cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', markersize=3)
plt.axhline(y=80, color='r', linestyle='--', label='80% variance')
plt.axhline(y=90, color='g', linestyle='--', label='90% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.title('PCA - How many components do we need?')
plt.legend()
plt.grid(True)
# plt.show()

# also print exact numbers
for threshold in [70, 80, 90]:
    n = np.argmax(cumulative_variance >= threshold) + 1
    # print(f"{threshold}% variance explained by {n} components")


# Apply PCA with 80% variance threshold
pca = PCA(n_components=0.80)
df_pca = pca.fit_transform(df_scaled)

# print(f"Components kept: {pca.n_components_}")
# print(f"Shape after PCA: {df_pca.shape}")






from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca)
    score = silhouette_score(df_pca, labels)
    silhouette_scores.append(score)
    # print(f"k={k}: silhouette score = {score:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Finding optimal number of clusters')
plt.grid(True)
# plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
# print(f"\nBest k: {best_k} with silhouette score: {max(silhouette_scores):.4f}")









from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ============================================================
# STEP 1: Try both KMeans and Agglomerative, find best k
# ============================================================

results = []

for k in range(2, 25):
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(df_pca)
    kmeans_score = silhouette_score(df_pca, kmeans_labels)
    results.append({'k': k, 'algorithm': 'KMeans', 'score': kmeans_score, 'labels': kmeans_labels})
    
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(df_pca)
    agg_score = silhouette_score(df_pca, agg_labels)
    results.append({'k': k, 'algorithm': 'Agglomerative', 'score': agg_score, 'labels': agg_labels})
    
    # print(f"k={k}: KMeans={kmeans_score:.4f}, Agglomerative={agg_score:.4f}")

# find best combination
best = max(results, key=lambda x: x['score'])
# print(f"\nBest: {best['algorithm']} with k={best['k']}, score={best['score']:.4f}")

# ============================================================
# STEP 2: Heuristic optimization - try different PCA components
# ============================================================

print("\n--- Heuristic Optimization ---")
best_overall_score = 0
best_overall_labels = None

for n_components in [0.70, 0.75, 0.80, 0.85, 0.90]:
    # refit PCA
    pca_temp = PCA(n_components=n_components)
    df_pca_temp = pca_temp.fit_transform(df_scaled)
    
    for k in range(2, 20):
        # try KMeans
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(df_pca_temp)
        score = silhouette_score(df_pca_temp, labels)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_labels = labels
            best_params = {'pca': n_components, 'k': k, 'algorithm': 'KMeans'}
        
        # try Agglomerative
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(df_pca_temp)
        score = silhouette_score(df_pca_temp, labels)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_labels = labels
            best_params = {'pca': n_components, 'k': k, 'algorithm': 'Agglomerative'}

# print(f"Best overall: {best_params}")
# print(f"Best silhouette score: {best_overall_score:.4f}")

# ============================================================
# STEP 3: Visualization
# ============================================================

# reduce to 2D just for visualization
pca_2d = PCA(n_components=2)
df_2d = pca_2d.fit_transform(df_scaled)

plt.figure(figsize=(12, 6))
scatter = plt.scatter(df_2d[:, 0], df_2d[:, 1], 
                      c=best_overall_labels, 
                      cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title(f"Clusters Visualization\n{best_params['algorithm']}, k={best_params['k']}, silhouette={best_overall_score:.4f}")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
# plt.show()

# ============================================================
# STEP 4: Generate submission CSV
# ============================================================

submission = pd.DataFrame({
    'Disease': disease_col.values,
    'cluster': best_overall_labels
})

submission.to_csv('submission.csv', index=False)
# print("\nSubmission saved!")
# print(submission['cluster'].value_counts())
# print(submission.head(10))







# read original disease names before any lowercasing
original_df = pd.read_csv('data.csv')
original_diseases = original_df['Disease'].reset_index(drop=True)

# create correct submission
# NEW - correct submission
original_df = pd.read_csv('data.csv')

submission = pd.DataFrame({
    'Disease': original_df['Disease'].values,
    'cluster': best_overall_labels
})

submission.to_csv('submission.csv', index=False)
print(f"Submission shape: {submission.shape}")
print(submission.head(10))

# verify shapes match
print(f"\nOriginal rows: {len(original_df)}")
print(f"Labels count: {len(best_overall_labels)}")

print(original_df['Disease'].value_counts()[original_df['Disease'].value_counts() > 1])




sample = pd.read_csv('sample_submission.csv')
print(f"Sample submission rows: {len(sample)}")
print(f"Our submission rows: {len(submission)}")

# check duplicates in sample
print(f"\nDuplicates in sample: {sample['Disease'].duplicated().sum()}")
print(f"Duplicates in ours: {submission['Disease'].duplicated().sum()}")







# for duplicate diseases, take the most common cluster assigned
submission_final = submission.groupby('Disease')['cluster'].agg(
    lambda x: x.mode()[0]
).reset_index()

# reorder to match sample submission order
sample = pd.read_csv('sample_submission.csv')
submission_final = sample[['Disease']].merge(submission_final, on='Disease', how='left')

# fill any diseases in sample that weren't in our data
submission_final['cluster'] = submission_final['cluster'].fillna(0).astype(int)

submission_final.to_csv('submission.csv', index=False)
print(f"Final shape: {submission_final.shape}")
print(f"Duplicates: {submission_final['Disease'].duplicated().sum()}")
print(submission_final.head(10))