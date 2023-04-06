import pandas as pd
import pickle
import streamlit as st
#import plotly.express as px


st.set_page_config(
    page_title="Revenue Grid Prediction",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
    
)


#Importing the CSV

data=pickle.load(open('csv_file.pkl','rb'))

a=[i.upper().replace('.','_') for i in data.columns]
data.rename(columns=dict(zip(data.columns,a)),inplace=True)

data_train=data.copy()

#Importing Model
rf=pickle.load(open('rf_model.pkl','rb'))


#Dropping Irrelevant Columns
data_train=data_train.drop(['POST_AREA','POST_CODE','REF_NO','YEAR_LAST_MOVED'],axis=1)


#Label Encoder
from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()

#Gender
data_train['GENDER'].replace(to_replace=['Male','Female','Unknown'],value=[1,2,0],inplace=True)

#Children
data_train['CHILDREN'].replace(to_replace=['Zero','1','2','3','4+'],value=[0,1,2,3,4],inplace=True)

b=[i for i in data_train.columns if data_train[i].dtypes=='object']
for i in b:
    data_train[i]=lc.fit_transform(data_train[i])

#################################################################################################################

#Streamlit Web Application

hide='''
<style>
Mainmenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
'''
st.markdown(hide,unsafe_allow_html=True)


st.write('# :orange[Revenue Grid Prediction] :')
st.markdown('#### This Application predicts the ***Revenue Grid*** of a ***Person*** :')

st.subheader('Enter Required Parameters :')
st.sidebar.subheader('Enter Required Parameters :')



col1,col2=st.columns(2)

q=[]
for i in data.columns:

    if data[i].dtypes=='float64':
        qq=st.sidebar.slider(i.replace('_',' '),10.0,1000.0,350.0)
        q.append(qq)

    elif i=='CHILDREN':
        with col1:
            qq=st.select_slider(i,options=data_train[i].sort_values().unique())
            q.append(qq)
    
    elif i=='TVAREA':
        with col1:
            qq=st.selectbox('TV AREA',data[i].sort_values().unique())
            q.append(qq)

    elif i=='REGION':
        with col1:
            qq=st.selectbox(i,data[i].sort_values().unique())
            q.append(qq)
    
    elif i in ['AGE_BAND','FAMILY_INCOME']:
        with col1:
            qq=st.select_slider(i.replace('_',' '),options=data[i].sort_values().unique())
            q.append(qq)

    elif i =='GENDER':
        with col1:
            qq=st.selectbox(i.replace('_',' '),data[i].sort_values().unique())
            q.append(qq)
    elif data[i].dtypes=='object':
        with col2:
            if i not in ['GENDER','CHILDREN','AGE_BAND','FAMILY_INCOME','POST_AREA','POST_CODE','TVAREA']:
                qq=st.selectbox(i.replace('_',' '),data[i].sort_values().unique())
                q.append(qq)



r=[i for i in data_train.columns if i!='REVENUE_GRID']
user_data=dict(zip(r,q))
user_data=pd.DataFrame(user_data,index=[0])


submit=st.button('Submit')

if submit:

    
    st.write('# Given Parameters :')
    st.write(user_data)

    #Label Encoding of Given Data

    #Gender
    user_data['GENDER'].replace(to_replace=['Male','Female','Unknown'],value=[1,2,0],inplace=True)

    #Remaining Columns

    b=[i for i in user_data.columns if user_data[i].dtypes=='object']
    for i in b:
        user_data[i]=lc.fit_transform(user_data[i])



    predict=rf.predict(user_data)
    st.write('# Predicted Revenue Grid :')
    predict=pd.DataFrame(predict,columns=['REVENUE'])
    predict['REVENUE']=predict['REVENUE'].replace({1:'Bad Revenue',2:'Good Revenue'})
    st.write(predict)

    predict_proba=rf.predict_proba(user_data)
    st.write('# Prediction Probability :')
    predict_proba=pd.DataFrame(predict_proba,columns=['Bad Revenue','Good Revenue'])
    st.write(predict_proba)


#File Uploader

st.subheader(':blue[Upload .CSV File :]')
upload=st.file_uploader('Upload the Dataset File',type='csv')

if upload is not None:

    st.success('File Uploaded Successfully')

    col3,col4=st.columns(2)

    with col3:
        proceed=st.button('Proceed')
    with col4:
        check=st.checkbox('Predicted Class Distribution Graph')


    if proceed:
        file_uploaded=pd.read_csv(upload)
        
        a=[i.upper().replace('.','_') for i in file_uploaded.columns]
        file_uploaded.rename(columns=dict(zip(file_uploaded.columns,a)),inplace=True)
        
        uploaded=file_uploaded.copy()

        #Dropping Irrelevant Columns
        uploaded=uploaded.drop(['POST_AREA','POST_CODE','REF_NO','YEAR_LAST_MOVED'],axis=1)

        st.write('# Given Dataset :')
        st.write(uploaded)


        #Label Encoding

        #Gender
        uploaded['GENDER'].replace(to_replace=['Male','Female','Unknown'],value=[1,2,0],inplace=True)

        #Children
        uploaded['CHILDREN'].replace(to_replace=['Zero','1','2','3','4+'],value=[0,1,2,3,4],inplace=True)

        b=[i for i in uploaded.columns if uploaded[i].dtypes=='object']
        for i in b:
            uploaded[i]=lc.fit_transform(uploaded[i])

        predict=rf.predict(uploaded)
        #st.write('# Predicted Revenue Grid :')
        predict=pd.DataFrame(predict,columns=['REVENUE'])

        predict_grid=predict.copy()
        predict_grid.rename(columns={'REVENUE':'REVENUE_GRID'},inplace=True)

        predict['REVENUE']=predict['REVENUE'].replace({1:'Bad Revenue',2:'Good Revenue'})
        #st.write(predict)
        
        col5,col6=st.columns(2)

        with col5:
            predict_proba=rf.predict_proba(uploaded)
            st.subheader('Prediction Probability :')
            predict_proba=pd.DataFrame(predict_proba,columns=['Bad Revenue','Good Revenue'])
            st.write(predict_proba)
        #with col6:
            #if check:
                #st.subheader('Predicted Class Distribution :')
                #fig=px.bar(predict,x=list(predict['REVENUE'].sort_values().unique()),y=predict['REVENUE'].value_counts().sort_values(),
                           #template='plotly_white',title='Class Spit',color=['1','2'])
                #st.plotly_chart(fig)


        st.write('# Predicted Dataset :')
        predicted=pd.concat([file_uploaded,predict,predict_grid],axis=1,ignore_index=False)
        st.dataframe(predicted)


        st.subheader('Download Predicted Dataset :')
        st.download_button('Download',data=predicted.to_csv(index=False),file_name='predicted_dataset.csv',mime='text/csv')
        
