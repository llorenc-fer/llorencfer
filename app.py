#-----librerías----------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import pickle 
import pickle_mixin
from sklearn.preprocessing import StandardScaler


#-----configuracion de página--------------------------------------------------------------------------

st.set_page_config(page_title='Basic Data Exploration: Titanic', layout='centered', page_icon='🛥️')

#-----cosas que podemos usar en toda nuestra app-------------------------------------------------------


df = pd.read_csv(r'C:\Users\lluri\Documents\Github\llorencfer\Titanic Streamlit\titaniccsv.csv')
df.drop('Cabin', inplace=True, axis=1)
df['Age'].fillna((df['Age'].mean()), inplace=True)
df = df.fillna(df['Embarked'].value_counts().index[0])

df.loc[df['Age']<=19, 'age_group'] = 'Teenage (<19)'
df.loc[df['Age'].between(20,29), 'age_group'] = 'Yong adult (20-29)'
df.loc[df['Age'].between(30,39), 'age_group'] = 'Adult (30-39)'
df.loc[df['Age'].between(40,49), 'age_group'] = 'Adult (40-49)'
df.loc[df['Age'].between(50,59), 'age_group'] = 'Older Adult (50-59)'
df.loc[df['Age']>=60, 'age_group'] = 'Elder (<60)'


#-----empieza la app-----------------------------------------------------------------------------------
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/TitanicBeken.jpg/1920px-TitanicBeken.jpg')
st.title('Case Study: Titanic Census ')
st.text("Our first Streamlit Data App")


#-----columnas-----------------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.write('Dataframe')
    st.dataframe(df)

with col2:
    st.write('Sex Distribution')
    proporcion = df["Sex"].value_counts().head()

    figcol = go.Figure(data=[go.Pie(labels=(proporcion/len(df * 100)).index, values=(proporcion/len(df * 100)).values, text=proporcion.index, hole=0.5)])
    figcol.update_layout(title = "Distribución por sexo", template = 'plotly_dark')
    st.plotly_chart(figcol)
   


#-----tablas que podemos usar--------------------------------------------------------------------------
#-----configuracion de tablas---------------------------------------------------------------------------
tabs = st.tabs(['Age', 'Survival Statistics','Age Group'])
#-----tabla 1-----------
tab_plots = tabs[0]
with tab_plots:
    st.write('Age distribution')
    fig0 = px.histogram(df, x ="Age", template='plotly_white', title="Edad de los pasajeros")
    fig0.update_layout(title="Age", xaxis_title="Age", yaxis_title="People", template = 'plotly_dark',)
    st.plotly_chart(fig0)
    
#-----tabla 2-----------
tab_plots = tabs[1]
with tab_plots:
    st.write("Survival per Sex: ")
    st.write("0 = No, 1 = Yes")
    fig1 = px.histogram(df, x='Sex', color='Survived')
    st.plotly_chart(fig1)

with tab_plots:
    fig1_1 = px.histogram(df, x='age_group', color='Survived',labels={
                     "age_group": "Age Group",
                     "Survived": "Survived",
                     },title="Survival per Age Group")
    st.plotly_chart(fig1_1)

with tab_plots:
    fig1_2 = px.histogram(df, x='Pclass', color='Survived',labels={
                     "Pclass": "Ticket class",
                     "Survived": "Survived",
                     },title="Survival per Ticket Class")
    st.plotly_chart(fig1_2)

    
#-----tabla 3-----------
tab_plots = tabs[2]
with tab_plots:
    st.write('Age Group Distribution')
    especies = df['age_group'].value_counts()
    fig2 = px.scatter_3d(x=especies, y=especies, z=especies, color=especies.index, template= 'plotly_dark',color_discrete_sequence=px.colors.qualitative.Alphabet)
    st.plotly_chart(fig2)


#SURVIVAL PREDICTION MODEL--------------------------------------------------------------------
st.title('Survival Prediction Model Web App')
st.write('Accuracy score is:  78,21%')
#loading the model
loaded_model = pickle.load(open('C:/Users\lluri/Documents/samplerepo/Upgrade Hub/Modulo 1/12-Scripts, APIs, Streamlit/Titanic Streamlit/trained_model_titanic.sav', 'rb'))
#creating a function for predicting

def titanic_prediction(input_data):
      
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    # print(input_data_reshaped)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person would not have survived'
    else:
        return 'The person would have survived'

def main():
    st.title('Testing the prediction model')

    #input data from user
    Pclass = st.selectbox("Insert ticket class",[2,1,0])
    Sex = st.selectbox('Insert Sex, 0 = male, 1 = female', [1,0])
    Age = st.slider("Insert Age",0,85,0)
    Sibsp = st.slider('Insert number of siblings/spouses', 0, 10, 1)
    Parch = st.slider('Insert number of parents/children', 0, 10, 1)
    Fare = st.number_input('Insert passenger fare')
    Embarked = st.selectbox('Insert Port of Embarkation (S:0, C:1, Q:2):', [2,1,0])

    #code for prediction
    survival_prediction = ''
    #creating a button for prediction
    if st.button('Titanic Survival Test Result'):
        survival_prediction = titanic_prediction([Pclass, Sex, Age, Sibsp, Parch, Fare, Embarked])

    st.success(survival_prediction)

if __name__ == '__main__':
    main()

#-----sidebar------------------------------------------------------------------------------------------
# st.set_option('deprecation.showPyplotGlobalUse', False) #para que no muestre warnings de versiones desfasadas
# st.sidebar.title('estos son algunos menus')
# st.sidebar.image('https://media.gettyimages.com/id/1310911999/es/foto/smoking-pipe-isolated-on-white-background.jpg?s=612x612&w=gi&k=20&c=G-4B2VqPafwGGgCa5bEPFV9jf6aao2xxXFE1wSrSv0Y=', width=100)
# st.sidebar.write('un texto')
# st.sidebar.write('---')
# st.sidebar.write('ootro texto')
# st.sidebar.write('---')
# if st.sidebar.button('Ver Dataframe'):
#     st.dataframe(df)
# if st.sidebar.button('Segundo click'):
#     st.write('Whoops! Algo salío mal')
#     st.image('https://media.gettyimages.com/id/1310911999/es/foto/smoking-pipe-isolated-on-white-background.jpg?s=612x612&w=gi&k=20&c=G-4B2VqPafwGGgCa5bEPFV9jf6aao2xxXFE1wSrSv0Y=', width=100)

# st.sidebar.slider('Slider sample', min_value=0,max_value=100)
# st.sidebar.checkbox('check sample', help='select values')
# st.sidebar.text_input(label='insert text')
# if(st.sidebar.button('botón de prueba')):
#     sns.set_theme(style='white')
#     sns.relplot(data=df, kind='scatter')
#     st.pyplot()
