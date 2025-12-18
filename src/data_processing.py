from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from src.data_ingestion import load_raw_data

def Data_processing():

    df = load_raw_data()
    x = df.drop(columns='Energy Consumption' ,axis= 1)
    y = df['Energy Consumption']

    # spliting data into independent and dependent column i.e X and y
    X_train , X_test ,y_train ,y_test = train_test_split(x , y ,train_size= 0.70 ,random_state=0)

    # Using encoding techxnique 
    categorical_column = x.select_dtypes(include='object').columns
    for i in categorical_column:
        le = LabelEncoder()
        X_train[i] = le.fit_transform(X_train[i])
        X_test[i] = le.transform(X_test[i])

    # Using Normalizing technique
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train ,X_test ,y_train ,y_test
