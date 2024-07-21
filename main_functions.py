import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 

from sklearn.model_selection import train_test_split



data_type = ['Categorical','Continuous']
Continuous = ['item_date', 'quantity tons', 'customer', 'country', 'application',
            'thickness', 'width', 'product_ref', 'delivery date', 'selling_price']
Categorical = ['id','item type','material_ref','status']

def import_data_and_preprocessing(df):
    df.isna().sum()
    categorical_data_type = "Categorical"
    continuous_data_type = "Continuous"

def get_fields_by_data_type(data_type):
        if(data_type == Categorical):
            return Categorical
        else:
            return Continuous  
    
def fillna_data(df):

    categorical_list = df.select_dtypes(include=['object', 'string']).columns
    continuous_list = df.select_dtypes(include=['int64','float64', 'float']).columns


   
    # fill Categorical null values
    model = LabelEncoder()
    for col in categorical_list:
        try:
            df[col]= model.fit_transform(df[col])
        except:
            pass
    #df['material_ref']= model.fit_transform(df['material_ref'])
            
    #df = pd.get_dummies(df,columns = ['status','item type'],dtype = 'int')
    #imputer = SimpleImputer(strategy='mean')
    #imputed_data = imputer.fit_transform(df[col])
       
    # fill continuous null values   
    for column in continuous_list:
        mean_value = df[column].mean()
        print("Mean Value :", mean_value )
        df[column].fillna(value=mean_value, inplace=True)

    return df




# Descriptive Statistics
def descriptive_statistics():
    pass

# Machine Learning
def machine_learning():
   
    
    pass


