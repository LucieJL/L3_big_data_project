import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import Birch, KMeans
from scipy import stats

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
df["classeAge"] = np.select(conditions, choices, default=0)

''' Encoding function (needs a list) '''
def labelEncoder(list_att):
    for att in list_att:
        le.fit(df[att].unique())
        df[att] = le.transform(df[att])

# Lead : Conversion en bit
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
#print(df[atts].isna().all(axis=1).value_counts())
#print(df[[col for col in df.columns if 'cps19_lead_int' in col]])
df[lead_trust_atts] = df[lead_trust_atts].isna().astype(int)
df[lead_int_atts] = df[lead_int_atts].isna().astype(int)
df[lead_strong_atts] = df[lead_strong_atts].isna().astype(int)

#print(df['cps19_demsat'].isna().value_counts())
#print(df['cps19_demsat'].unique())
'''print(df[[col for col in df.columns if 'cps19_party_member' in col]].isna().all(axis=1).value_counts())
print(df[[col for col in df.columns if 'cps19_party_member' in col]])'''
#print()
#exit()
#print(df['cps19_party_member'].isna().value_counts())
#print(df['cps19_province'].isna().value_counts())
#print(df['cps19_spend_educ'].isna().value_counts())
#print(df['cps19_spend_env'].isna().value_counts())
#print(df['cps19_spend_just_law'].isna().value_counts())
#print(df['cps19_spend_defence'].isna().value_counts())
#print(df['cps19_spend_imm_min'].isna().value_counts())
#print(df['cps19_spend_imm_min'].unique())
df.dropna(subset=['cps19_prov_id'], inplace=True)
df.dropna(subset=['cps19_vote_2015'], inplace=True)


# Education : Numérisation/Ordonnement
df['cps19_education'] = df['cps19_education'].replace({'No schooling':0, 'Some elementary school':1, 'Completed elementary school':2,'Some secondary/ high school': 3, 'Completed secondary/ high school': 4, 'Some technical, community college, CEGEP, College Classique': 5, 'Completed technical, community college, CEGEP, College Classique': 6, 'Some university': 7, "Bachelor's degree": 8, "Master's degree":9, 'Professional degree or doctorate': 10, "Don't know/ Prefer not to answer": -1})

#print(df[df['cps19_education'] == -1])
edu = np.array(df['cps19_education'])
edu_mean = int(np.mean(edu[edu > -1]))
#print(np.mean(edu[edu>-1]))
#print(np.median(edu[edu>-1]))
#df[df['cps19_education'] == -1]['cps19_education'] = int(np.mean(edu[edu > -1]))
df.loc[df['cps19_education'] == -1, ['cps19_education']] = edu_mean

# Emploi : Remplacement de "je ne sais pas" par "autre"
df['cps19_employment']= df['cps19_employment'].replace({"Don't know/ Prefer not to answer": 'Other'})
#print(df['cps19_employment'])
# Religion : Groupement par "grand courant regligieux"
#Non religieux/oriental/juif/musulman/chretiens/chretiens déviré/other
df['cps19_religion']=df['cps19_religion'].replace({"None/ Don't have one/ Atheist":'Non religieux', 'Agnostic':'Non religieux', 'Buddhist/ Buddhism':'oriental', 'Hindu':'oriental', 'Jewish/ Judaism/ Jewish Orthodox':'juif', 'Muslim/ Islam': 'musulman', 'Sikh/ Sikhism':'oriental', "Anglican/ Church of England":'chretiens', 'Baptist':'chretiens', 'Catholic/ Roman Catholic/ RC':'chretiens', "Greek Orthodox/ Ukrainian Orthodox/ Russian Orthodox/ Eastern Orthodox": 'chretiens', "Jehovah's Witness":'chretiens', 'Lutheran':'chretiens', "Mormon/ Church of Jesus Christ of the Latter Day Saints":'chretiens déviré', 'Pentecostal/ Fundamentalist/ Born Again/ Evangelical':'chretiens déviré','Presbyterian':'chretiens','Protestant':'chretiens', 'United Church of Canada':'chretiens', 'Christian Reformed':'chretiens', 'Salvation Army':'chretiens', 'Mennonite':'chretiens déviré', 'Other (please specify)':'other',"Don't know/ Prefer not to answer":'other'})
#print(df['cps19_religion'])
#df['row_num'] = df.reset_index().index


#Genre rien à faire

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
print(df['cps19_pos_cannabis'].unique())

brc = Birch(n_clusters=8)
brc.fit(df[to_cluster])
df['cps19_pos'] = brc.predict(df[to_cluster])


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


''' Sélection des attributs '''

attributes = [
    'classeAge',
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
    'cps19_demsat', # bof
    'cps19_pos',
    'label'
] + lead_strong_atts + lead_int_atts + lead_trust_atts

df = df[attributes]

for attr in [
    'cps19_gender',
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
]:
    le = preprocessing.LabelEncoder()
    le.fit(df[attr].unique())
    df[attr] = le.transform(df[attr])


''' Séparation du dataset de test '''

dfTest = df[df.index.isin(testIndexes[0])]
dfTrain = df[~df.index.isin(testIndexes[0])]

# Retirer les ~1000 individus sans reponses (1226)
dfTrain = dfTrain[dfTrain['label'] != '']


''' chi square si besoin
crosstab = pd.crosstab(dfTrain['cps19_province'], dfTrain['label'])

print(stats.chi2_contingency(crosstab))
exit(0)
'''

''' Entrainement '''

def detailPrint(y_test,y_test_pred):
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_test_pred))
    print("Exactitude :", accuracy_score(y_test, y_test_pred))
    print("Précision :", precision_score(y_test, y_test_pred, average='macro'))
    print("Rappel :", recall_score(y_test, y_test_pred, average='macro'))
    print("F1-score :", f1_score(y_test, y_test_pred, average='macro'))
    

catNB = CategoricalNB()
multNB = MultinomialNB()
rf = RandomForestClassifier()

lst_catNB_accuracy=[]
lst_multNB_accuracy=[]
lst_rf_accurancy=[]

index_partition=0

# Utilisation de "k-fold cross validation"
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(dfTrain):
    print("TRAIN:", train_index, "TEST:", test_index)

    X_train = dfTrain.iloc[train_index].loc[:, dfTrain.columns != 'label']
    #X_train = dfTrain.drop(columns=['label','row_num' ]).iloc[train_index, :]
    y_train = dfTrain.iloc[train_index]['label']
    X_test = dfTrain.iloc[test_index].loc[:, dfTrain.columns != 'label']
    #X_test = dfTrain.drop(columns=['label','row_num' ]).iloc[test_index, :]
    y_test = dfTrain.iloc[test_index]['label']

    #CategoricalNB
    print('CategorialNB')
    catNB.fit(X_train, y_train)
    y_test_pred = catNB.predict(X_test)
    lst_catNB_accuracy.append(accuracy_score(y_test, y_test_pred))
    detailPrint(y_test,y_test_pred)
   
    # RandomForestClassifier
    print("RandomForestClassifier")
    rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)
    lst_rf_accurancy.append(accuracy_score(y_test, y_test_pred))
    detailPrint(y_test,y_test_pred)
    
    # Classificateur naïf de bayes
    # https://www.stat4decision.com/fr/foret-aleatoire-avec-python/

    # Arbre de décision (Random Forest)
    # https://www.cours-gratuit.com/tutoriel-python/tutoriel-python-matriser-la-classification-nave-baysienne-avec-scikit-learn

    # K plus proches voisins
    # https://medium.com/@kenzaharifi/bien-comprendre-lalgorithme-des-k-plus-proches-voisins-fonctionnement-et-impl%C3%A9mentation-sur-r-et-a66d2d372679
    index_partition=index_partition+1

print('--CategorialNB-- ')
print('max exactitude : ' + str(max(lst_catNB_accuracy)))
print('mean exactitude : ' + str(sum(lst_catNB_accuracy) / len(lst_catNB_accuracy)))
print("--RandomForestClassifier--")
print('max exactitude : ' + str(max(lst_rf_accurancy)))
print('mean exactitude : ' + str(sum(lst_rf_accurancy) / len(lst_rf_accurancy)))

''' Test de l'algorithme '''

# Classificateur naïf de bayes

#Création de dp de sortie avec liste tampon pour les indexs puis ajoute prediction et création du fichier .txt
#Avec l'algo catNB
'''df_sortie=pd.DataFrame()
lst=[]
lst=dfTest['row_num'].tolist()
df_sortie["row_num"] = lst
df_sortie["label"]=catNB.predict(dfTest.drop(columns=['label','row_num']))
df_sortie.to_csv('prediction.txt', index=False, sep="\t", header=False)'''

# Arbre de décision (Random Forest)
# K plus proches voisins

