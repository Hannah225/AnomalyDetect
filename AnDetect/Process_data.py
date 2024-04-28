import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
from datetime import datetime
import umap
import torch

#for a fixed train test split we use this split
error_sys = [
            34, 59, 5, 68, 37, 2, 67, 52, 54, 65, 15, 30, 47,  #many errors in these systems
            31, 902, 27, 12, #Not so many errors
            28, 49, 44, 36, 64, 70, 41, 51, 903, 46, 26, 23, 72, 16 # very little errors
        ]

#We can get the data in various different preprocessed states:
#All data is returned in the same format from all functions (yay!)

# get_normal_data: noc scaling, no transformations
# get_scaled_robust_data: robust scaling
# get_scaled_standard_data: standard scaling
# get_pca_data: data dimensions is reduced with pca; number of dimensions is def. in n_components param
# get_umap_data: dimension reduction with umap; also n_components param
# get_merk_data: error data coupled with merk (all noted as error)
# get_gauss_data: errors calculated with gaussian filter and certain threshold
# mlp_data: convert the results of all other functions to tensors for the mlp
# get_AnomalyPred: returns anomaly and likelihood predicitions

def get_data():
    data = pd.read_csv("data/new/probabilistic/predictions.csv")
    data = data.rename({'sto':'error'}, axis=1)
    data = data.rename({'day':'date'}, axis=1)
    data["likelihood"] = [float(val[7:-1]) for val in data["likelihood"].to_list()]
    return data

def train_test_split(data, error_sys = error_sys):
    #do the train_test_split:
    train_df = data[data['system'].isin(error_sys)]
    test_df = data[~data['system'].isin(error_sys)]
    return train_df, test_df

def rand_train_test_split(data, seed, error_sys = error_sys):
    np.random.seed(seed)
    #do the train_test_split:
    all_systems = data["system"].unique()
    train_systems = np.random.choice(all_systems, size=24, replace=False)

    train_df = data[data['system'].isin(train_systems)]
    test_df = data[~data['system'].isin(train_systems)]
    return train_df, test_df


#return normal data (with train/test split):
def get_normal_data(seed, error_sys = error_sys):
    data = get_data()

    #do the train_test_split:
    train_df, test_df = rand_train_test_split(data, seed)

    #split x and y:
    train_x = train_df.values[:,-256:].astype(float)
    test_x = test_df.values[:,-256:].astype(float)

    train_y = train_df["error"].values == 1
    test_y = test_df["error"].values == 1

    return train_df, test_df, train_x, test_x, train_y, test_y


def get_scaled_standard_data(seed, error_sys = error_sys):
    scaler = StandardScaler()
    train_df, test_df, train_x, test_x, train_y, test_y = get_normal_data(seed, error_sys)
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return train_df, test_df, train_x, test_x, train_y, test_y


def get_scaled_robust_data(seed, error_sys = error_sys):
    scaler = RobustScaler() #apparently works well with data that contains outliers
    train_df, test_df, train_x, test_x, train_y, test_y = get_normal_data(seed, error_sys)
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return train_df, test_df, train_x, test_x, train_y, test_y

def get_pca_data(seed, error_sys = error_sys):
    #n_components tells us to how many dimensions the pca is reducing the data to
    #Initialize the pca
    n_components = 2
    pca = PCA(n_components=n_components)
    #get the data
    data = get_data()
    #transform the data and then replace the old data with the new pca tranformed data
    pca_res = pca.fit_transform(data.values[:,-256:])
    pca_res = pd.DataFrame(pca_res)
    data = pd.concat([data.iloc[:, :-256], pca_res], axis=1)

    #do the train test split:
    train_df, test_df = rand_train_test_split(data, seed)

    #split x and y:
    train_x = train_df.values[:,-n_components:].astype(float)
    test_x = test_df.values[:,-n_components:].astype(float)

    train_y = train_df["error"].values == 1
    test_y = test_df["error"].values == 1
    
    return train_df, test_df, train_x, test_x, train_y, test_y

def get_umap_data(seed, error_sys = error_sys):
    #takes about 30sec to transform
    #function has same logic as pca transformation
    n_components = 2
    umap_transformation = umap.UMAP(n_neighbors=15, n_components=n_components, min_dist=0.1)

    data = get_data()
    umap_transformation.fit(data.values[:,-256:])
    transformed = umap_transformation.transform(data.values[:,-256:])

    transformed = pd.DataFrame(transformed)
    data = pd.concat([data.iloc[:, :-256], transformed], axis=1)

    #do the train test split:
    train_df, test_df = rand_train_test_split(data, seed)

    #split x and y:
    train_x = train_df.values[:,-n_components:].astype(float)
    test_x = test_df.values[:,-n_components:].astype(float)

    train_y = train_df["error"].values == 1
    test_y = test_df["error"].values == 1
    
    return train_df, test_df, train_x, test_x, train_y, test_y


def get_merk_data(seed, error_sys = error_sys):
    data = get_data()
    #einfach alle beide zusammengenommen und immer als Fehler gewertet
    data['error'] = data.apply(lambda row: 1 if row['error'] == 1 or row['merk'] == 1 else 0, axis=1)
    #do the train_test_split:
    train_df, test_df = rand_train_test_split(data, seed)

    #split x and y:
    train_x = train_df.values[:,-256:].astype(float)
    test_x = test_df.values[:,-256:].astype(float)

    train_y = train_df["error"].values == 1
    test_y = test_df["error"].values == 1

    return train_df, test_df, train_x, test_x, train_y, test_y

##### gauss helper functions ###########################

def get_dates(idx, system_data, system_firstIndex):
    corr_dateInd = []
    corr_dateInd.append(idx)
    curr_idx = idx
    while(
        (curr_idx > system_firstIndex) and
        (((datetime.strptime(system_data['date'][curr_idx], '%Y-%m-%d') - 
          datetime.strptime(system_data['date'][curr_idx - 1], '%Y-%m-%d')).days == 1))
    ):
        corr_dateInd.append(curr_idx)
        curr_idx = curr_idx - 1
    curr_idx = idx
    while(
        (curr_idx < system_firstIndex + len(system_data)-1) and
        (((datetime.strptime(system_data['date'][curr_idx + 1], '%Y-%m-%d')
            - datetime.strptime(system_data['date'][curr_idx], '%Y-%m-%d')).days  == 1))
    ):
        corr_dateInd.append(curr_idx)
        curr_idx = curr_idx + 1
    return corr_dateInd

def gauss_dist(corr_days, idx, sigma):
    # Generate the Gaussian distribution for each element in 'days' based on its distance from 'error_index'
    const = (1/(sigma * np.sqrt(2*np.pi)))
    gaussian_dist = [
        const * np.exp(-0.5 * (np.square(day - idx) / sigma)) for day in corr_days
    ]

    return gaussian_dist

def apply_gaussian_filter(df, sigma):
    df['gauss_error'] = np.zeros(shape=(len(df),1))
    for system in df['system'].unique():
        system_firstIndex = df[df['system'] == system].index.min() #startet bei 0
        system_data = df[df['system'] == system] #Indizierung folgt den indizies des gesamten df

        #geht von system_firstIndex bis system_firstIndex + len(system_data) - 1
        error_indices = system_data.index[system_data['error'] == 1].tolist()
        #print(system, system_firstIndex, (system_data['date'][system_firstIndex] - system_data['date'][system_firstIndex + 1]).days)
        for idx in error_indices:
            corr_days = get_dates(idx, system_data, system_firstIndex)
            corr_days = list(set(corr_days))
            corr_days.sort()
            dist = gauss_dist(corr_days, idx, sigma = sigma)
            #print(34, len(dist), len(corr_days), (corr_days[0] - 1) - corr_days[-1])
            start = corr_days[0]
            end = corr_days[-1]
            df.loc[start:end, 'gauss_error'] += dist
    df.loc[df['error'], 'gauss_error'] = 1
    return df

#### end gauss helper functions #####

def get_gauss_data(seed, error_sys = error_sys):
    data = get_data()
    #gaußschen Filter draufwerfen (s.o.)
    data = apply_gaussian_filter(data, 0.5)
    #cutoff für die error werte festlegen, dann error basierend auf den cutoff werten berechnen
    cutoff_val = 0.5
    data['error'] = data.apply(lambda row: 1 if row['error'] == 1 or row['gauss_error'] >= cutoff_val else 0, axis=1)
    data.drop(columns = ['gauss_error'], axis = 1) #drop gauss error col
    #do the train_test_split:
    train_df, test_df = rand_train_test_split(data, seed)

    #split x and y:
    train_x = train_df.values[:,-256:].astype(float)
    test_x = test_df.values[:,-256:].astype(float)

    train_y = train_df["error"].values == 1
    test_y = test_df["error"].values == 1

    return train_df, test_df, train_x, test_x, train_y, test_y


def mlp_data(train_x, test_x, train_y, test_y):
    x_tensor =  torch.from_numpy(train_x).float() #torch.Size([24589, 257])
    y_tensor =  torch.from_numpy(train_y).float() # torch.Size([24589, 1])
    y_tensor = y_tensor.unsqueeze(1) #Adds one dimension to the tensor to make them comparable
    #CREATE TRAIN DATASET
    train_ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)

    #CREATE TEST DATASET
    xtest_tensor =  torch.from_numpy(test_x).float() #torch.Size([13235, 257])
    ytest_tensor =  torch.from_numpy(test_y).float() 
    ytest_tensor = ytest_tensor.unsqueeze(1)
    test_ds =  torch.utils.data.TensorDataset(xtest_tensor, ytest_tensor)

    return train_ds, test_ds

def get_AnomalyPred(train_df, test_df,):
    train_anomaly_scores = train_df["anomaly_score"].values
    train_likelihoods = train_df["likelihood"].values

    test_anomaly_scores = test_df["anomaly_score"].values
    test_likelihoods = test_df["likelihood"].values

    train_y = train_df["error"].values == 1

    #get best cutoff values for anomaly and likelihood
    p,r,t = precision_recall_curve(train_y, train_anomaly_scores)
    p[p==0] = 1e-10
    r[r==0] = 1e-10
    f1 = 2*p*r/(p+r)
    anomaly_threshold = t[np.argmax(f1)]

    p,r,t = precision_recall_curve(train_y, train_likelihoods)
    p[p==0] = 1e-10
    r[r==0] = 1e-10
    f1 = 2*p*r/(p+r)
    likelihood_threshold = t[np.argmax(f1)]

    anomaly_score_predictions = test_anomaly_scores > anomaly_threshold
    likelihood_predictions = test_likelihoods > likelihood_threshold

    return anomaly_score_predictions, likelihood_predictions