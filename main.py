import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.cluster import Birch, KMeans
from point_h import *

''' Paramétrage '''

df = pd.read_csv('CES19.csv')
testIndexes = pd.read_csv('exemple.txt', sep='\t', header=None)


''' Pipeline de prétraitrement '''

# Age : Division des classes d'âges existantes en fonction de leur médiane (réduit la dimension quand encodé)
col = "cps19_age"
choices = [1,2,3,4,5,6]
conditions = [(df[col]>=18) & (df[col]<29),
            (df[col]>=29) & (df[col]<35),
            (df[col]>=35) & (df[col]<45),
            (df[col]>=45) & (df[col]<55),
            (df[col]>=55) & (df[col]<66),
            (df[col]>=66)]
df["cps19_age"] = np.select(conditions, choices, default=0)

# Emploi : Remplacement de "je ne sais pas" par "autre" (réduction d'une dimension)
df['cps19_employment']= df['cps19_employment'].replace({"Don't know/ Prefer not to answer": 'Other (please specify)'})

# Religion : Groupement par "grand courant regligieux"
#Nonreligious (Nonrelig) / Eastern religions (EasternRelig) / Jew / Muslim / Christian / Extremism (ChristianExt)
df['cps19_religion']=df['cps19_religion'].replace({
    "None/ Don't have one/ Atheist":'Nonrelig', 'Agnostic':'Nonrelig', 
    'Buddhist/ Buddhism':'EasternRelig', 'Hindu':'EasternRelig', 'Sikh/ Sikhism':'EasternRelig',
    'Jewish/ Judaism/ Jewish Orthodox':'Jew', 
    'Muslim/ Islam': 'Muslim',  
    "Anglican/ Church of England":'Christian', 'Baptist':'Christian', 'Catholic/ Roman Catholic/ RC':'Christian', 
    "Greek Orthodox/ Ukrainian Orthodox/ Russian Orthodox/ Eastern Orthodox": 'Christian', 
    "Jehovah's Witness":'Christian', 'Lutheran':'Christian', 
    "Mormon/ Church of Jesus Christ of the Latter Day Saints":'ChristianExt', 
    'Pentecostal/ Fundamentalist/ Born Again/ Evangelical':'ChristianExt',
    'Presbyterian':'Christian','Protestant':'Christian', 'United Church of Canada':'Christian', 
    'Christian Reformed':'Christian', 'Salvation Army':'Christian', 
    'Mennonite':'ChristianExt', 'Other (please specify)':'Other',
    "Don't know/ Prefer not to answer":'Other'})

#Genre + éducation rien à faire (si ce n'est encoder)

# Ajout du numéro de ligne de départ dans un attribut
df['row_num'] = df.index

# Lead : Conversion en booléen (si valeur existe =1, sinon =0)
lead_trust_atts = [
    'cps19_lead_trust_113', 'cps19_lead_trust_114',
    'cps19_lead_trust_115', 'cps19_lead_trust_116', 'cps19_lead_trust_117',
    'cps19_lead_trust_118', 'cps19_lead_trust_119', 'cps19_lead_trust_120'
]
lead_int_atts = [
    'cps19_lead_int_113', 'cps19_lead_int_114',
    'cps19_lead_int_115', 'cps19_lead_int_116', 'cps19_lead_int_117',
    'cps19_lead_int_118', 'cps19_lead_int_119', 'cps19_lead_int_120'
]
lead_strong_atts = [
    'cps19_lead_strong_113', 'cps19_lead_strong_114',
    'cps19_lead_strong_115', 'cps19_lead_strong_116', 'cps19_lead_strong_117',
    'cps19_lead_strong_118', 'cps19_lead_strong_119', 'cps19_lead_strong_120'
]
# Cast en int
df[lead_trust_atts] = df[lead_trust_atts].isna().astype(int)
df[lead_int_atts] = df[lead_int_atts].isna().astype(int)
df[lead_strong_atts] = df[lead_strong_atts].isna().astype(int)

# Définition des attributs à encoder
attributes_to_encode = [
    'cps19_gender',
    'cps19_education',
    'cps19_employment',
    'cps19_religion',
    #'cps19_prov_id',
    #'cps19_vote_2015',
    'cps19_fed_id',
    'cps19_spend_educ',
    'cps19_spend_env',
    'cps19_spend_just_law',
    'cps19_spend_defence',
    'cps19_spend_imm_min',
    'cps19_province', # bof
    'cps19_bornin_canada', # bof
    'cps19_children', # bof
    'cps19_marital', # bof
    'cps19_union', # bof
    'cps19_sexuality', # bof
    'cps19_demsat',
]

labelEncoder(df, attributes_to_encode)

predictAttributeNaiveBayes(df, [
    'cps19_age', 'cps19_employment', 'cps19_religion', 'cps19_fed_id'
] + lead_strong_atts + lead_trust_atts + lead_int_atts, 'cps19_prov_id')

labelEncoder(df, ['cps19_prov_id'])


predictAttributeNaiveBayes(df, [
    'cps19_age', 'cps19_employment', 'cps19_religion', 'cps19_fed_id', 'cps19_prov_id'
] + lead_strong_atts + lead_trust_atts + lead_int_atts, 'cps19_vote_2015')

labelEncoder(df, ['cps19_vote_2015'])

attributes_to_encode.extend(['cps19_prov_id', 'cps19_vote_2015'])


''' Fusionner les étiquettes pour avoir une seule colonne label df['label']'''

label_cols = [
    'cps19_votechoice',
    'cps19_votechoice_pr',
    'cps19_vote_unlikely',
    'cps19_vote_unlike_pr',
    'cps19_v_advance',
]
df[label_cols] = df[label_cols].fillna('')
df['label'] = df['cps19_votechoice'] + df['cps19_votechoice_pr'] + df['cps19_vote_unlikely'] + df['cps19_vote_unlike_pr'] + df['cps19_v_advance']

''' Clustering des attributs ordonnables '''

conversion_attributes = {
    'Strongly disagree': 0,
    'Somewhat disagree': 1,
    'Neither agree nor disagree': 2,
    'Somewhat agree': 3,
    'Strongly agree': 4,
    "Don't know/ Prefer not to answer": 2,
    np.nan: 2
}

to_cluster = [
    'cps19_pos_cannabis',
    'cps19_pos_life',
    'cps19_pos_fptp',
    'cps19_pos_carbon',
    'cps19_pos_energy',
    'cps19_pos_envreg',
    'cps19_pos_jobs',
    'cps19_pos_subsid',
    'cps19_pos_trade'
]
df[to_cluster] = df[to_cluster].applymap(lambda x: conversion_attributes[x])

brc = Birch(n_clusters=8)
brc.fit(df[to_cluster])
df['cps19_pos'] = brc.predict(df[to_cluster])


''' Attributes selection and enconding '''

'''attributes_to_encode = [
    'cps19_gender',
    'cps19_education',
    'cps19_employment',
    'cps19_religion',
    'cps19_prov_id',
    'cps19_vote_2015',
    'cps19_fed_id',
    'cps19_spend_educ',
    'cps19_spend_env',
    'cps19_spend_just_law',
    'cps19_spend_defence',
    'cps19_spend_imm_min',
    'cps19_province', # bof
    'cps19_bornin_canada', # bof
    'cps19_children', # bof
    'cps19_marital', # bof
    'cps19_union', # bof
    'cps19_sexuality', # bof
    'cps19_demsat',
]'''

attributes_not_to_encode = [
    'row_num',
    'cps19_age',
    'cps19_pos',
    'label'
] + lead_strong_atts + lead_int_atts + lead_trust_atts

df = df[attributes_to_encode + attributes_not_to_encode]


''' Séparation du dataset de test '''

dfTest = df[df.index.isin(testIndexes[0])]
dfTrain = df[~df.index.isin(testIndexes[0])]

# Retirer les ~1000 individus sans reponses (1226)
dfTrain = dfTrain[dfTrain['label'] != '']


''' Entrainement '''

nb_classifiers = []
rf_classifiers = []

nb_accuracies = []
nb_balanced_accuracies = []
rf_accuracies = []
rf_balanced_accuracies = []

index_partition = 0


# Utilisation de "k-fold cross validation"

    
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(dfTrain):
    print("TRAIN:", train_index, "TEST:", test_index)

    X_train = dfTrain.drop(columns=['label', 'row_num']).iloc[train_index, :]
    y_train = dfTrain.iloc[train_index]['label']
    X_test = dfTrain.drop(columns=['label', 'row_num']).iloc[test_index, :]
    y_test = dfTrain.iloc[test_index]['label']

    #CategoricalNB
    print('CategorialNB')
    nb = CategoricalNB()
    nb_classifiers.append(nb)
    nb.fit(X_train, y_train)
    y_test_pred = nb.predict(X_test)
    nb_accuracies.append(accuracy_score(y_test, y_test_pred))
    nb_balanced_accuracies.append(balanced_accuracy_score(y_test, y_test_pred))
    printMetrics(y_test, y_test_pred)
   
    # RandomForestClassifier
    print("RandomForestClassifier")
    rf = RandomForestClassifier()
    rf_classifiers.append(rf)
    rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)
    rf_accuracies.append(accuracy_score(y_test, y_test_pred))
    rf_balanced_accuracies.append(balanced_accuracy_score(y_test, y_test_pred))
    printMetrics(y_test, y_test_pred)

    index_partition += 1

print('\nCategorialNB')
printFinalMetrics(nb_accuracies, nb_accuracies)
print("\nRandomForestClassifier")
printFinalMetrics(rf_accuracies, rf_accuracies)


''' Prédiction du dataset de test '''

# Récupération du meilleur classificateur RandomForest durant le KFold
best_rf = rf_classifiers[rf_accuracies.index(max(rf_accuracies))]

# Création du dataframe de sortie, prédiction des labels et sauvegarde en CSV
df_sortie = pd.DataFrame()
df_sortie["row_num"] = dfTest['row_num'].tolist()
df_sortie["label"] = best_rf.predict(dfTest.drop(columns=['label', 'row_num']))
df_sortie.to_csv('prediction.txt', index=False, sep="\t", header=False)



## Java's garbage collector 
# Education : Numérisation/Ordonnement
#edu = np.array(df['cps19_education'])
#edu_mean = int(np.mean(edu[edu > -1]))
#df['cps19_education'] = df['cps19_education'].replace({'No schooling':0, 'Some elementary school':1, 'Completed elementary school':2,'Some secondary/ high school': 3, 'Completed secondary/ high school': 4, 'Some technical, community college, CEGEP, College Classique': 5, 'Completed technical, community college, CEGEP, College Classique': 6, 'Some university': 7, "Bachelor's degree": 8, "Master's degree":9, 'Professional degree or doctorate': 10, "Don't know/ Prefer not to answer": edu_mean})



