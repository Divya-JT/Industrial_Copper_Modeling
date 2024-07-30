import streamlit as st
import pandas as pd
from main_functions import *
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import os
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
from prediction_feature import *
import pickle


# ALGORITHMS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# EVALUATION METRICS
from sklearn.metrics import accuracy_score, auc,precision_score,recall_score,f1_score,roc_auc_score, roc_curve,auc
from sklearn.metrics import mean_squared_error
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
    tab1, tab2 = st.tabs(['PREDICT SELLING PRICE', 'PREDICT STATUS'])
    with tab1:
        try:
            selling_price = prediction.regression()
            st.write(selling_price)

            if selling_price:
                # apply custom css style for prediction text
                #style_prediction()
                st.markdown(f'### <div class="center-text">Predicted Selling Price = {selling_price}</div>', unsafe_allow_html=True)
                st.balloons()
        

        except ValueError:
            col1,col2,col3 = st.columns([0.26,0.55,0.26])
            with col2:
                st.warning('##### Quantity Tons / Customer ID is empty')

    with tab2:
        try:
            status = prediction.classification()
            if status == 1:
                # apply custom css style for prediction text
                #style_prediction()
                st.markdown(f'### <div class="center-text">Predicted Status = Won</div>', unsafe_allow_html=True)
                st.balloons()
                

            elif status == 0:
                
                # apply custom css style for prediction text
                #style_prediction()
                st.markdown(f'### <div class="center-text">Predicted Status = Lost</div>', unsafe_allow_html=True)
                st.snow()
        
        except ValueError:
            col1,col2,col3 = st.columns([0.15,0.70,0.15])
            with col2:
                st.warning('##### Quantity Tons / Customer ID / Selling Price is empty')


    pass




def show_evaluation_data(type, train: any, test:any):
    #st.write(f"Train:{train} Test : {test}")
    data = [{'Train': train, 'Test':test}]

    st.header(fr"**:orange[{type}]**  data evaluation")
    col1, col2 = st.columns([1,6])
    with col1: 
        st.subheader(r"**Train data:**")
    with col2:
        st.subheader(fr"***:orange[{train}]***")

    col1, col2 = st.columns([1,6])
    with col1: 
        st.subheader(r"**Test data:**")
    with col2:
        st.subheader(fr"***:orange[{test}]***")
    




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

def show_Skewness():
    #skewness_and_normalization()    
    option = st.selectbox(
    "Select Continuous data from the list ",
    ['quantity tons', 'customer', 'country', 'application',
    'thickness', 'width', 'product_ref', 'selling_price'])
    
    if option:
       
        st.session_state.industrial_copper_data[option] = pd.to_numeric(st.session_state.industrial_copper_data[option], errors='coerce')

        skew_result = st.session_state.industrial_copper_data[option].skew()
        col1, col2 = st.columns([0.45,1])
        col1.subheader(r"**Skewness Result**")
        col2.subheader(fr"***{skew_result}***")            
        #st.write(skew_result)  

        plt.hist(skew_result, bins=20)
        #sns.boxplot(skew_result)
        # Display the plot in Streamlit
        st.pyplot()
        col_data = st.session_state.industrial_copper_data[option]     
        
        #st.write(col_data)  
        st.subheader("Normlized Data")
        normalize_data = (col_data-col_data.mean())/(col_data.max()-col_data.min())
        #fig = px.histogram(col_data, x=option)
        #st.plotly_chart(figure_or_data=fig)
        
        plt.hist(normalize_data, bins=20)
        # Display the plot in Streamlit
        st.pyplot()

    pass

def show_descriptive_statistics():
    st.write("Data  Distribution")

    option = st.selectbox(
    "Select data from the list ",
    ['quantity tons', 'customer', 'country', 'application',
    'thickness', 'width', 'product_ref', 'selling_price', 'status', 'item type'])
    
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

#Machine learning
def show_machine_learning():
    #df = st.session_state.industrial_copper_data
    #df.drop('material_ref'),axis = 1, inplace = True
    algorithm_options = [
    'Linear Regression',
    'Logistic Regression',
    'KNeighbors Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'Gradient Boosting Regressor'
    ]
    evaluation_matrics_options = ['accuracy_score','precision_score','recall_score','f1_score','roc_auc_score', 'roc_curve']

    # Create the selectbox widget with a label and options
    selected_algorithm = st.selectbox('Select an Algorithm', algorithm_options)
    #selected_evaluation_metrics = st.selectbox('Select an Evaluation Metrics', evaluation_matrics_options)
    
    model_map = {
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(),
    'KNeighbors Regressor': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
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

   
    threshold = 0.5
    train_prediction = (train_prediction > threshold).astype(int)
    test_prediction = (test_prediction > threshold).astype(int)
    #y_true = st.session_state.industrial_copper_data["selling_price"]
    #mse = mean_squared_error(y_binary, test_prediction)

    #accuracy_score
    show_evaluation_data("accuracy_score", accuracy_score(y_train,train_prediction), accuracy_score(y_test,test_prediction))
    st.divider()
    # precision_score
    show_evaluation_data("precision_score", precision_score(y_train,train_prediction), precision_score(y_test,test_prediction))
    st.divider()
    #recall_score
    show_evaluation_data("recall_score", recall_score(y_train,train_prediction), recall_score(y_test,test_prediction))
    st.divider()
    # f1_score
    show_evaluation_data("f1_score", f1_score(y_train,train_prediction), f1_score(y_test,test_prediction))
    st.divider()
    # roc_auc_score
    show_evaluation_data("roc_auc_score", roc_auc_score(y_train,train_prediction), roc_auc_score(y_test,test_prediction))
    st.divider()
    # roc_curve
    show_evaluation_data("roc_curve", roc_curve(y_train,train_prediction), roc_curve(y_test,test_prediction))
    st.divider()
    pass


def categorical_data():
   
    algorithm_options = [
    'KNeighbors Classifier',
    'Decision Tree Classifier',
    'Random Forest Classifier',
    'Gradient Boosting Classifier'
    ]
    evaluation_matrics_options = ['accuracy_score','precision_score','recall_score','f1_score']

    # Create the selectbox widget with a label and options
    selected_algorithm = st.selectbox('Select an Algorithm', algorithm_options)
    #selected_evaluation_metrics = st.selectbox('Select an Evaluation Metrics', evaluation_matrics_options)
    
    model_map = {
    'KNeighbors Classifier': KNeighborsClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier()
    }

    # Get the model class based on the selected algorithm name
    model_class = model_map[selected_algorithm] 

    final_df = pd.read_csv("final_data.csv")
    df_c = final_df.copy()
    # filter the status column values only 1 & 0 rows in a new dataframe ['Won':1 & 'Lost':0]
    df_c = df_c[(df_c.status == 1) | (df_c.status == 0)]
    x = df_c.drop('status', axis=1)
    y = df_c['status']
       
    x_new, y_new = SMOTETomek().fit_resample(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)
    model = model_class
    model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    


     #accuracy_score
    show_evaluation_data("accuracy_score", accuracy_score(y_train,train_prediction), accuracy_score(y_test,test_prediction))
    st.divider()
    # precision_score
    show_evaluation_data("precision_score", precision_score(y_train,train_prediction), precision_score(y_test,test_prediction))
    st.divider()
    #recall_score
    show_evaluation_data("recall_score", recall_score(y_train,train_prediction), recall_score(y_test,test_prediction))
    st.divider()
    # f1_score
    show_evaluation_data("f1_score", f1_score(y_train,train_prediction), f1_score(y_test,test_prediction))
    st.divider()
    
    
    
pass

if select == "Home":
    show_home_screen()
    pass
elif select == "Skewness":
    show_Skewness()
elif select == "Descriptive Statistics":
    show_descriptive_statistics()
    
elif select == "Machine Learning":
    tab1, tab2 = st.tabs(['Regression', 'Classification'])
    with tab1:
        st.header("Regression")
        show_machine_learning()
    with tab2:
        st.header("Classification")
        categorical_data()
elif select == "Prediction":
    show_prediction()
    pass

primaryColor = "#F63366"
backgroundColor = "black"