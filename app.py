#-----librerías----------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import pickle 
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
from unicodedata import name 


#-----configuracion de página--------------------------------------------------------------------------

st.set_page_config(page_title='Basic Data Exploration: Titanic', layout='centered', page_icon='🛥️')

#-----cosas que podemos usar en toda nuestra app-------------------------------------------------------

url = 'https://raw.githubusercontent.com/llorenc-fer/llorencfer/main/titaniccsv.csv'
df = pd.read_csv(url, index_col=0)
df.drop('Cabin', inplace=True, axis=1)
df.drop('PassengerId', inplace=True, axis=1)
df['Age'].fillna((df['Age'].mean()), inplace=True)
df = df.fillna(df['Embarked'].value_counts().index[0])

#Create column with age group
df.loc[df['Age']<=19, 'age_group'] = 'Teenager (<19)'
df.loc[df['Age'].between(20,29), 'age_group'] = 'Young adult (20-29)'
df.loc[df['Age'].between(30,39), 'age_group'] = 'Adult (30-39)'
df.loc[df['Age'].between(40,49), 'age_group'] = 'Adult (40-49)'
df.loc[df['Age'].between(50,59), 'age_group'] = 'Older Adult (50-59)'
df.loc[df['Age']>=60, 'age_group'] = 'Elder (>60)'
#Create column with latitude and longitude
df.loc[df['Embarked']==0, 'lon'] = '50.896364'
df.loc[df['Embarked']==1, 'lon'] = '51.850910'
df.loc[df['Embarked']==2, 'lon'] = '49.648194'

df.loc[df['Embarked']==0, 'lat'] = '-1.406013'
df.loc[df['Embarked']==1, 'lat'] = '-8.294143'
df.loc[df['Embarked']==2, 'lat'] = '-1.612260'




#-----empieza la app-----------------------------------------------------------------------------------
st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/TitanicBeken.jpg/1920px-TitanicBeken.jpg')
st.text("RMS Titanic off the Isle of Wight, based on photograph by Frank Beken of Cowes")
st.title('Case Study: Titanic Census ')
st.text("Our first Streamlit Data App")

st.write('Dataframe Overview')
st.dataframe(df)
#-----columnas-----------------------------------------------------------------------------------------
# col1, col2 = st.columns(2)

# with col1:
#     st.write('Dataframe')
#     st.dataframe(df)
# with col2:
    

#-----tablas que podemos usar--------------------------------------------------------------------------
#-----configuracion de tablas---------------------------------------------------------------------------
tabs = st.tabs(['General Data Exploring', 'Survival Statistics','Survival Prediction Model'])
#-----tabla 1-----------
# tab_plots = tabs[0]
# with tab_plots:
#     col1, col2,col3 = st.columns(3)
#     with col1:
#         pass
#     with col2:
tab_plots = tabs[0]

with tab_plots:        
    st.image('https://cdn.activestate.com/wp-content/uploads/2019/08/exploratory-data-analysis-using-python-blog-hero.jpg')
    st.text("Image from activestate.com")
    st.title('General Data Exploring')
with tab_plots:    
    st.write('Sex Distribution')
    proporcion = df["Sex"].value_counts().head()

    figcol = go.Figure(data=[go.Pie(labels=(proporcion/len(df * 100)).index, values=(proporcion/len(df * 100)).values, text=proporcion.index, hole=0.5)])
    figcol.update_layout(title = "Distribución por sexo", template = 'plotly_dark')
    st.plotly_chart(figcol)

with tab_plots:        
    st.write('Age and Sex distribution')
    fig0 = px.histogram(df, x ="Age", color='Sex', template='plotly_white', title="Edad de los pasajeros") #marker_color=colors)
    fig0.update_layout(title="Distribuición por edad y sexo", xaxis_title="Age", yaxis_title="People", template = 'plotly_dark',)
    st.plotly_chart(fig0)

with tab_plots:
    st.write('Age Group Distribution')
    especies = df['age_group'].value_counts()
    #fig2 = px.scatter_3d(df, x=especies, y=especies, z=especies, color=especies.index, template= 'plotly_dark',color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig2 = px.treemap(especies, path=[especies.index], values=especies, height=700, title='Tamaño de los grupos de edades', color_discrete_sequence = px.colors.qualitative.Dark2)
    st.plotly_chart(fig2)

with tab_plots:
    
    # st.write('Passenger Origin Distribution')
    # sizearray = np.asarray(df['lon'].value_counts())
    # latarray = np.asarray(df['lat'].unique())
    # lonarray = np.asarray(df['lon'].unique())
    # sizeref = 2.*max(sizearray)/(40.**2)

    # fig = go.Figure(data=go.Scattergeo(lon=lonarray,
    #                                lat=latarray,
    #                                mode='markers',
    #                                marker=dict(
    #                                    size=sizearray,
    #                                    sizemode='area',
    #                                    sizeref=sizeref,
    #                                    sizemin=4,
    #                                    color=[
    #                                        'rgb(93, 164, 214)',
    #                                        'rgb(255, 144, 14)',
    #                                        'rgb(44, 160, 101)'
    #                                    ]
    #                                )))

    # fig.update_layout(autosize=True, height=600, title='Shipping Ports', geo_scope='europe')
    # fig.update_geos(fitbounds="locations") 
    # fig.show()
    st.write('Port embarking by density')
    html = open("pruebamapa1.html", "r", encoding='utf-8').read()
    st.components.v1.html(html,height=600)
    #components.html("<html><center><h1>Hello, World</h1></center></html>", width=200, height=200)
    
    #components.iframe(r"C:\Users\lluri\Documents\samplerepo\Upgrade Hub\Modulo 1\12-Scripts, APIs, Streamlit\Titanic Streamlit\pruebamapa1.html", width=600, height=450)

#-----tabla 2-----------
tab_plots = tabs[1]
with tab_plots:        
    st.image('https://lithub.com/wp-content/uploads/sites/3/2021/02/titanic-feat1.jpg')
    st.text('Image from Lithub.com')
    st.title('Survival Statistics')

with tab_plots:
    st.write("Survival per Sex: ")
    st.write("Supervivencia en función del género")
    fig1 = px.histogram(df, x='Sex', color='Survived',text_auto=True)
    series_names1 = ["No", "Yes"]
    for idx, name in enumerate(series_names1):
        fig1.data[idx].name = name
        fig1.data[idx].hovertemplate = name
    st.plotly_chart(fig1)

with tab_plots:
    st.write('Survival per Age')
    fig3 = px.histogram(df, x="Age", color="Survived")
    series_names3 = ["No", "Yes"]
    for idx, name in enumerate(series_names3):
        fig3.data[idx].name = name
        fig3.data[idx].hovertemplate = name
    fig3.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 10
    )
)
    fig3.update_layout(
    yaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 25
    )
)
    st.plotly_chart(fig3)

with tab_plots:
    fig1_1 = px.histogram(df, x='age_group', color='Survived',labels={
                     "age_group": "Age Group",
                     "Survived": "Survived",
                     },title="Survival per Age Group",text_auto=True).update_xaxes(categoryorder='total descending')
    series_names1_1 = ["No", "Yes"]
    for idx, name in enumerate(series_names1_1):
        fig1_1.data[idx].name = name
        fig1_1.data[idx].hovertemplate = name
    
    st.plotly_chart(fig1_1)

with tab_plots:
    fig1_2 = px.histogram(df, x='Pclass', color='Survived',labels={
                     "Pclass": "Ticket class",
                     "Survived": "Survived",
                     },title="Survival per Ticket Class",text_auto=True)
    fig1_2.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [1, 2, 3],
        ticktext = ['First Class', 'Second Class', 'Third Class']
    )
)
    series_names1_2 = ["No", "Yes"]
    for idx, name in enumerate(series_names1_2):
        fig1_2.data[idx].name = name
        fig1_2.data[idx].hovertemplate = name
    st.plotly_chart(fig1_2)
   
#-----tabla 3-----------
tab_plots = tabs[2]
with tab_plots:
        #SURVIVAL PREDICTION MODEL--------------------------------------------------------------------
    st.title('Survival Prediction Model Web App')
    st.write('Accuracy score is:  78,21%')
    st.image('https://data-science-blog.com/wp-content/uploads/2018/07/deep-learning-header-1030x352.png')
    st.write('Image from data-science-blog.com')

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
        st.write('Would this person have survived?')
        #input data from user
        Pclass = st.selectbox("Insert ticket class",[2,1,0])
        Sex = st.selectbox('Insert Sex: 0 = male, 1 = female', [1,0])
        Age = st.slider("Insert Age",0,85,0)
        Sibsp = st.slider('Insert number of siblings/spouses', 0, 10, 1)
        Parch = st.slider('Insert number of parents/children', 0, 10, 1)
        Fare = st.number_input('Insert passenger fare')
        Embarked = st.selectbox('Insert Port of Embarkation (Southampton: 0, Cherbourg Place: 1, Queenstown: 2):', [2,1,0])

        #code for prediction
        survival_prediction = ''
        #creating a button for prediction
        if st.button('Titanic Survival Test Result'):
            survival_prediction = titanic_prediction([Pclass, Sex, Age, Sibsp, Parch, Fare, Embarked])

        st.success(survival_prediction)

    if __name__ == '__main__':
        main()



# menu = option_menu(

#     menu_title=None,
#     options=["Introducción", "Dataframe", "Análisis"],
#     icons= ["house", "list", "clipboard-plus"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
#     styles="dark"
# if menu =="Introducción":
#     st.title("INTRODUCCIÓN")
#     ....
#     ....


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

## cd 'C:\Users\lluri\Documents\samplerepo\Upgrade Hub\Modulo 1\12-Scripts, APIs, Streamlit\Titanic Streamlit'
## streamlit run app.py


