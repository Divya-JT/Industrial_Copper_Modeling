import streamlit as st
import pandas as pd
from main_functions import *
from sklearn.preprocessing import LabelEncoder 
import os
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 


# ALGORITHMS
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# EVALUATION METRICS
from sklearn.metrics import accuracy_score, auc,precision_score,recall_score,f1_score,roc_auc_score, roc_curve,auc

st.set_option('deprecation.showPyplotGlobalUse', False)




def initialize_session_state():
    if 'industrial_copper_data' not in st.session_state:
        st.session_state.industrial_copper_data = None

def main():
    initialize_session_state()
    


# Page configuration

st.set_page_config (layout="wide", page_title="Industrial Copper Modelling")
#st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"ColorMeBlue text”"}</h1>', unsafe_allow_html=True)

with st.sidebar:
    select = option_menu("Industrial Copper Modelling", ["Home","Skewness","Descriptive Statistics","Machine Learning", "Prediction"])

def show_prediction():
    pass

def show_evaluation_data(type, train: any, test:any):
    st.write(f"Train:{train} Test : {test}")
    data = [{'Train': train, 'Test':test}]

    st.subheader(f"\"{type}\" data evaluation")
    col1, col2 = st.columns([1,1])
    with col1: 
        st.write("**Train data:**")
    with col2:
        st.write(train)

    col1, col2 = st.columns([1,1])
    with col1: 
        st.write("**Test data:**")
    with col2:
        st.write(test)
    

    df = pd.DataFrame(data={'Train':train, 'Test':test}, index=[type])
    #df = df.astype(float)
    #pd.set_option('display.float_format', lambda x: '%f')
    st.markdown(df.to_markdown())
    #st.write(df.dtypes)



def show_home_screen():
    st.title ("Industrial Copper Modelling")
    uploaded_file = st.file_uploader("Drag & Drop or Select JSON file here to upload Industrial Copper Modelling data.", type=["csv"])
    upload_btn = st.button(label = "Upload the file", type = "primary", disabled = uploaded_file == None)
    if upload_btn:
        with st.spinner("Loading the data"):
            st.session_state['industrial_copper_data'] = pd.read_csv(uploaded_file)
            st.session_state.industrial_copper_data.iloc[:1]["country"] = None
            st.write(st.session_state['industrial_copper_data'])
            st.session_state.industrial_copper_data = fillna_data(st.session_state.industrial_copper_data)
            st.write("After pre processing")
            st.write(st.session_state.industrial_copper_data)
            st.success("Done")
        
        
pass


if select == "Home":
    show_home_screen()
    pass
elif select == "Skewness":
    #skewness_and_normalization()    
    
    option = st.selectbox(
    "Select Continuous data from the list ",
    ['quantity tons', 'customer', 'country', 'application',
    'thickness', 'width', 'product_ref', 'selling_price'])
    
    if option:
       
        st.session_state.industrial_copper_data[option] = pd.to_numeric(st.session_state.industrial_copper_data[option], errors='coerce')

        skew_result = st.session_state.industrial_copper_data[option].skew()
        st.write(skew_result)

        plt.hist(skew_result, bins=20)
        #sns.boxplot(skew_result)
        # Display the plot in Streamlit
        st.pyplot()



       
        
        col_data = st.session_state.industrial_copper_data[option]     
        st.write(col_data)  
        normalize_data = (col_data-col_data.mean())/(col_data.max()-col_data.min())
        
        
        fig = px.histogram(col_data, x=option)
        st.plotly_chart(figure_or_data=fig)
        plt.hist(normalize_data, bins=20)
        # Display the plot in Streamlit
        st.pyplot()

       
    
    pass

elif select == "Descriptive Statistics":
    descriptive_statistics()
    st.write("Data  Distribution")

    option = st.selectbox(
    "Select data from the list ",
    ['quantity tons', 'customer', 'country', 'application',
    'thickness', 'width', 'product_ref', 'selling_price', 'status', 'item type', 'material_ref'])
    
    fig = px.histogram(st.session_state.industrial_copper_data[option], nbins=20)
    fig.update_layout(
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            xaxis_title_font=dict(color='#0DF0D4'),
            yaxis_title_font=dict(color='#0DF0D4')
        )
    fig.update_traces(hoverlabel=dict(bgcolor="#0E1117"),
                            hoverlabel_font_color="#0DF0D4")
    fig.update_xaxes(title_text=option)
    fig.update_yaxes(title_text="id")
    fig.update_traces(marker_color='#1BD4BD')
    st.plotly_chart(fig, theme=None, use_container_width=True)

    pass
elif select == "Machine Learning":
    machine_learning()
    #df = st.session_state.industrial_copper_data
    #df.drop('material_ref'),axis = 1, inplace = True
    algorithm_options = [
    'Logistic Regression',
    'KNeighbors Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'Gradient Boosting Regressor'
    ]
    evaluation_matrics_options = ['accuracy_score','precision_score','recall_score','f1_score','roc_auc_score', 'roc_curve']

    # Create the selectbox widget with a label and options
    selected_algorithm = st.selectbox('Select an Algorithm', algorithm_options)
    selected_evaluation_metrics = st.selectbox('Select an Evaluation Metrics', evaluation_matrics_options)
    
    model_map = {
    'Logistic Regression': LogisticRegression(),
    'KNeighbors Regressor': KNeighborsRegressor(),
    'Decision Tree Regressorr': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor()
    }

    # Get the model class based on the selected algorithm name
    model_class = model_map[selected_algorithm]

    df = st.session_state.industrial_copper_data
    X = df.drop(['selling_price'], axis = 1)
    X = X.drop(['quantity tons'], axis = 1)
    y = df ['selling_price']
    threshold = 0.3
    y_binary = (y >= threshold).astype(int)  # Convert to 0 or 1 based on the threshold

    x_train, x_test, y_train, y_test = train_test_split(X,y_binary,test_size = 0.25)


    #print("Machine learning === x :", X, " y : ", y)
    #st.write(X)
    #st.write(y)
    model = model_class
    model.fit(X,y_binary)
    plt.figure(figsize=(10,10))
    plt.show()

    model.fit(x_train,y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)


    train_data: any = None
    test_data: any = None
    if(selected_evaluation_metrics == "accuracy_score"):
        train_data = accuracy_score(y_train,train_prediction)
        test_data = accuracy_score(y_test,test_prediction)

    elif(selected_evaluation_metrics == "precision_score"):
        train_data = precision_score(y_train,train_prediction)
        test_data = precision_score(y_test,test_prediction)

    elif(selected_evaluation_metrics == "recall_score"):
        train_data = recall_score(y_train,train_prediction)
        test_data = recall_score(y_test,test_prediction)

    elif(selected_evaluation_metrics == "f1_score"):
        train_data = f1_score(y_train,train_prediction)
        test_data = f1_score(y_test,test_prediction)

    elif(selected_evaluation_metrics == "roc_auc_score"):
        train_data = roc_auc_score(y_train,train_prediction) 
        test_data = roc_auc_score(y_test,test_prediction)

    elif(selected_evaluation_metrics == "roc_curve"):   
        train_data = roc_curve(y_train,train_prediction)  
        test_data = roc_curve(y_test,test_prediction)

        #fpr,tpr,_ = roc_curve(y_train,train_prediction)
        #roc_auc = auc(fpr,tpr)
        #plt.plot(fpr,tpr)

    show_evaluation_data(selected_evaluation_metrics, train_data, test_data)
    

    pass


elif select == "Prediction":
    pass

primaryColor = "#F63366"
backgroundColor = "black"