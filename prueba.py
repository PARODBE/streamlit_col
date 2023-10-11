import streamlit as st
import numpy as np
import pandas as pd
#from streamlit_ketcher import st_ketcher
from PIL import Image
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.neighbors import KNeighborsRegressor
import random
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_validate, KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer, mean_absolute_error
import time




st.image('juntas.png', use_column_width=False,width=300)


#molfile = st_ketcher(molecule_format="MOLFILE")
#st.markdown("molfile:")
#st.code(molfile)

progress_text = "Loading transporter datasets"
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()


def fingerprints_inputs(dataframe):

    X=np.array([AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048,useFeatures=True) for mol in [Chem.MolFromSmiles(m) for m in list(dataframe.canonical_smiles)]])
    y=dataframe.pchembl_value.astype('float')
    return X,y

df_bcrp=PandasTools.LoadSDF("chembl_data_bcrp.sdf")
df_bcrp['inchi']=[Chem.MolToInchi(m) for m in df_bcrp.ROMol]
df_bcrp=df_bcrp[df_bcrp.standard_type=='IC50']
df_bcrp['pchembl_value']=df_bcrp['pchembl_value'].astype('float')
df_mrp2=PandasTools.LoadSDF("chembl_data_mrp2.sdf")
df_mrp2['inchi']=[Chem.MolToInchi(m) for m in df_mrp2.ROMol]
df_mrp2=df_mrp2[df_mrp2.standard_type=='IC50']
df_mrp2['pchembl_value']=df_mrp2['pchembl_value'].astype('float')
df_mrp3=PandasTools.LoadSDF("chembl_data_mrp3.sdf")
df_mrp3['inchi']=[Chem.MolToInchi(m) for m in df_mrp3.ROMol]
df_mrp3=df_mrp3[df_mrp3.standard_type=='IC50']
df_mrp3['pchembl_value']=df_mrp3['pchembl_value'].astype('float')
df_mrp4=PandasTools.LoadSDF("chembl_data_mrp4.sdf")
df_mrp4['inchi']=[Chem.MolToInchi(m) for m in df_mrp4.ROMol]
df_mrp4=df_mrp4[df_mrp4.standard_type=='IC50']
df_mrp4['pchembl_value']=df_mrp4['pchembl_value'].astype('float')
df_oat1=PandasTools.LoadSDF("chembl_data_OATP1b1.sdf")
df_oat1['inchi']=[Chem.MolToInchi(m) for m in df_oat1.ROMol]
df_oat1=df_oat1[df_oat1.standard_type=='IC50']
df_oat1['pchembl_value']=df_oat1['pchembl_value'].astype('float')
df_oat2=PandasTools.LoadSDF("chembl_data_OATP1b3.sdf")
df_oat2['inchi']=[Chem.MolToInchi(m) for m in df_oat2.ROMol]
df_oat2=df_oat2[df_oat2.standard_type=='IC50']
df_oat2['pchembl_value']=df_oat2['pchembl_value'].astype('float')
df_bsep=PandasTools.LoadSDF("chembl_data_bsep.sdf")
df_bsep['inchi']=[Chem.MolToInchi(m) for m in df_bsep.ROMol]
df_bsep=df_bsep[df_bsep.standard_type=='IC50']
df_bsep['pchembl_value']=df_bsep['pchembl_value'].astype('float')
df_pgp=PandasTools.LoadSDF("chembl_data_pgp.sdf")
df_pgp['inchi']=[Chem.MolToInchi(m) for m in df_pgp.ROMol]
df_pgp=df_pgp[df_pgp.standard_type=='IC50']
df_pgp['pchembl_value']=df_pgp['pchembl_value'].astype('float')

st.write('<p style="color:green; font-size:24px;">&#10003; Datasets uploaded successfully</p>', unsafe_allow_html=True)


progress_text = "Building Low-Level models"
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

X_bcrp,y_bcrp=fingerprints_inputs(df_bcrp)
X_mrp2,y_mrp2=fingerprints_inputs(df_mrp2)
X_mrp3,y_mrp3=fingerprints_inputs(df_mrp3)
X_mrp4,y_mrp4=fingerprints_inputs(df_mrp4)
X_oat1,y_oat1=fingerprints_inputs(df_oat1)
X_oat2,y_oat2=fingerprints_inputs(df_oat2)
X_bsep,y_bsep=fingerprints_inputs(df_bsep)
X_pgp,y_pgp=fingerprints_inputs(df_pgp)

random.seed(46)

model_bcrp=RandomForestRegressor(**{'criterion': 'squared_error', 'max_depth': None, 'min_samples_split': 2, 'n_estimators': 16},random_state=46).fit(X_bcrp,y_bcrp)

model_mrp2=RandomForestRegressor(**{'criterion': 'squared_error', 'max_depth': None, 'min_samples_split': 3, 'n_estimators': 16},random_state=46).fit(X_mrp2,y_mrp2)

model_mrp3=SVR(**{'C': 1, 'gamma': 0.001, 'kernel': 'linear'}).fit(X_mrp3,y_mrp3)

model_mrp4=RandomForestRegressor(**{'criterion': 'squared_error', 'max_depth': None, 'min_samples_split': 3, 'n_estimators': 80},random_state=46).fit(X_mrp4,y_mrp4)

model_oat1=RandomForestRegressor(**{'criterion': 'squared_error', 'max_depth': 2, 'min_samples_split': 5, 'n_estimators': 16},random_state=46).fit(X_oat1,y_oat1)

model_oat2=SVR(**{'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}).fit(X_oat2,y_oat2)

model_bsep=XGBRegressor(**{'colsample_bytree': 1, 'max_depth': 2, 'min_child_weight': 2, 'n_estimators': 16}).fit(X_bsep,y_bsep)

model_pgp=RandomForestRegressor(**{'criterion': 'squared_error', 'max_depth': 2, 'min_samples_split': 2, 'n_estimators': 16},random_state=46).fit(X_pgp,y_pgp)

# Número de divisiones para las validaciones cruzadas
n_splits = 5

# Número de repeticiones del proceso de validación cruzada
n_repeats = 2
st.write('<p style="color:green; font-size:24px;">&#10003; LLM built successfully</p>', unsafe_allow_html=True)

cv_bcrp=cross_val_score(model_bcrp,X_bcrp,y_bcrp,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))
cv_mrp2=cross_val_score(model_mrp2,X_mrp2,y_mrp2,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))
cv_mrp3=cross_val_score(model_mrp3,X_mrp3,y_mrp3,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))
cv_mrp4=cross_val_score(model_mrp4,X_mrp4,y_mrp4,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))
cv_oat1=cross_val_score(model_oat1,X_oat1,y_oat1,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))
cv_oat2=cross_val_score(model_oat2,X_oat2,y_oat2,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))
cv_bsep=cross_val_score(model_bsep,X_bsep,y_bsep,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))
cv_pgp=cross_val_score(model_pgp,X_pgp,y_pgp,cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats),scoring=make_scorer(mean_absolute_error))


arrays2=[cv_pgp, cv_bcrp,cv_oat1,cv_oat2, cv_mrp4, cv_mrp2,cv_bsep, cv_mrp3]

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.rcParams["font.family"] = 'Franklin Gothic Medium'
plt.rcParams.update({'font.size': 25})

# Set custom x-axis tick labels
x_labels = ['P-gp','BCRP', 'OATP1B1', 'OATP1B3','MRP4','MRP2', 'BSEP','MRP3']

# Create a dataframe for plotting
data = []
for i, array in enumerate(arrays2):
    data.extend([(x_labels[i], value) for value in array])

df_ = pd.DataFrame(data, columns=['Transporter', 'pIC50'])

# Plot the violinplot with Seaborn
fig=plt.figure(figsize=(10, 12))
sns.violinplot(x='pIC50', y='Transporter', data=df_, inner='stick')

# Set x-axis tick labels
plt.yticks(range(len(x_labels)), x_labels)

plt.xlabel('$pIC_{50}$')
plt.ylabel('')
# Show the plot
st.pyplot(fig)


arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)

# Título de la aplicación
st.title("Upload file")

# Caja de carga de archivos
archivo = st.file_uploader("Select one file", type=["csv", "txt", "xlsx", "sdf"], accept_multiple_files=True)

# # Comprobar si se ha cargado un archivo
# if archivo is not None:
#     st.write(f"File name: {archivo.name}")
#     st.write(f"File type: {archivo.type}")

#     # Mostrar el contenido del archivo
#     if st.button("Show content"):
#         contenido = archivo.read()
#         st.write("Content file:")
#         st.code(contenido)







