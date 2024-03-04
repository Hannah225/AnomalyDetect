import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
import torch

def get_data():
    """ Takes pickel data (faults.pkl) and returns full dataset

        Parameters
        ----------
        None, searchs for data in given directory

        Returns
        ------
        merged dataset with all variables an binary errorcodes
        """
    faults_df = pd.read_pickle("data/faults.pkl") 

    #Read in the latent variables
    column_names = ["index", "date", "system"]
    latent_mean = pd.read_csv("data/output_latent_means.csv")
    latent_var = pd.read_csv("data/output_latent_variances.csv")


    #Rename the columns (add suffix mean/var and rename the first three columns)
    num_remaining_columns = latent_var.shape[1] - len(column_names)
    mean_cols = [f'mean_{i+1}' for i in range(num_remaining_columns)]
    var_cols = [f'var_{i+1}' for i in range(num_remaining_columns)]
    latent_mean.columns = column_names + mean_cols
    latent_var.columns = column_names + var_cols

    faults_df = faults_df.rename(columns={'day': 'date'})
    faults_df['system'] = faults_df['system'].str.lstrip('0')
    faults_df['system'] = faults_df['system'].astype(int)

    #Full outer join of all three data frames
    latent_merge = pd.merge(latent_mean, latent_var, on = ['date', 'system'])
    latent_merge_full = pd.merge(latent_merge, faults_df, on = ['date', 'system'], how = 'outer')

    #create new column with boolean on wether there was an error or not
    latent_merge_full['error'] = np.where(latent_merge_full['errorcodes'].notna(), True, False)
    #drop unneccessary columns
    latent_merge_full = latent_merge_full.drop(columns=['index_x', 'index_y', 'errorcodes'])

    return latent_merge_full

def get_new_data():
    data = pd.read_csv("AnDetect/data/new/ensemble/predictions.csv")
    #data = pd.read_csv("AnDetect/data/new/probabilistic/predictions.csv")
    data = data.rename({'sto':'error'}, axis=1)
    data = data.rename({'day':'date'}, axis=1)
    data = data.drop(columns = ['Unnamed: 0', 'merk', 'anomaly_score', 'likelihood'], axis = 1)

    return data

def get_AnomalyScorePred():
    data = pd.read_csv("AnDetect/data/new/ensemble/predictions.csv")
    data["likelihood"] = [float(val) for val in data["likelihood"].to_list()]

    error_sys = [
            34, 59, 5, 68, 37, 2, 67, 52, 54, 65, 15, 30, 47,  #many errors in these systems
            31, 902, 27, 12, #Not so many errors
            28, 49, 44, 36, 64, 70, 41, 51, 903, 46, 26, 23, 72, 16 # very little errors
        ]

    #RANDOM Train Test split (for testing purposes only)
    #all_systems = data["system"].unique()
    #error_sys = np.random.choice(all_systems, size=24, replace=False)

    #Do the train test split
    train_data = data[data['system'].isin(error_sys)]
    test_data = data[~data['system'].isin(error_sys)]

    test_anomaly_scores = test_data["anomaly_score"].values
    test_likelihoods = test_data["likelihood"].values
    test_sto = test_data["sto"].values == 1

    train_anomaly_scores = train_data["anomaly_score"].values
    train_likelihoods = train_data["likelihood"].values
    train_sto = train_data["sto"].values == 1

    #get best cutoff values for anomaly and likelihood
    p,r,t = precision_recall_curve(train_sto, train_anomaly_scores)
    p[p==0] = 1e-10
    r[r==0] = 1e-10
    f1 = 2*p*r/(p+r)
    anomaly_threshold = t[np.argmax(f1)]

    p,r,t = precision_recall_curve(train_sto, train_likelihoods)
    p[p==0] = 1e-10
    r[r==0] = 1e-10
    f1 = 2*p*r/(p+r)
    likelihood_threshold = t[np.argmax(f1)]

    anomaly_score_predictions = test_anomaly_scores > anomaly_threshold
    likelihood_predictions = test_likelihoods > likelihood_threshold

    return anomaly_score_predictions, likelihood_predictions


def dataset_info(dataset, full_dataset, name, gaussian_data = False, error_thresh = 0): #not accurate for gaussian data
    """ Takes dataset and parent dataset and print out infomration about it

        Parameters
        ----------
        dataset to be described, the parent dataset of that set and the name of the dataset (i.e. Training)
        rest is optinal

        Returns
        ------
        Prints
        """
    if(gaussian_data):
        num_faults = [0, 0]
        num_faults[1] = [1 if x > error_thresh else 0 for x in dataset['gauss_error']].count(1)
        num_faults[0] = len(dataset) - num_faults[1]
    else:
        num_faults = dataset['error'].value_counts(1)
    total_length = len(dataset)
    balance = num_faults[0] / total_length
    anteil = len(dataset) / len(full_dataset)
    #Print out Inofrmation about the Dataset
    print(f"Dataset: {name}")
    print(f"Partition: {anteil:.2%}")
    print(f"Total Length: {total_length}")
    #print(f"Errors: Ussing Error-Threshold {error_thresh}") #only needed for gaussian data
    print(f"Number of Errors: {num_faults[1]}")
    print(f"Number of Non-errors: {num_faults[0]}")
    print(f"Percent of non-errors: {balance:.2%}\n")


def data_preprocess(df, gaussian_data = False, error_thresh = 0):
    """ Takes full dataset and retruns test/train split as well as tensor datasets.
        Gaussian Data ist set to False, if set to True function will assume that 
        gaussian data is used - not implemented yet

        Parameters
        ----------
        df : full dataframe (pd df)

        Note: Function Prints out information about the datasets

        Returns
        ------
        train_ds and test_ds: tensor datasets of test and train data
        test_y: original prediciton data with test train split to evaluate the model
        """

    #TEST TRAIN SPLIT
    #Systeme mit den meisten Fehlern (diese sollten im Training sein):
    #Hier einige Systeme mit vielen Fehlern nehmen, einige mit wenigen. Der Großteil der Fehler ist im Trainingsset
    error_sys = [
        34, 59, 5, 68, 37, 2, 67, 52, 54, 65, 15, 30, 47,  #many errors in these systems
        31, 902, 27, 12, #Not so many errors
        28, 49, 44, 36, 64, 70, 41, 51, 903, 46, 26, 23, 72, 16 # very little errors
    ]

    #partition data based on wether the system is in error sys or not
    train_data = df[df['system'].isin(error_sys)]
    test_data = df[~df['system'].isin(error_sys)]

    dataset_info(df, df, "All Data", gaussian_data, error_thresh)
    dataset_info(train_data, df, "Trainingdata" , gaussian_data, error_thresh)
    dataset_info(test_data, df, "Testdata", gaussian_data, error_thresh)

    # if(gaussian_data):
    #     train_x = train_data.iloc[:, 2:-3]
    #     test_x = test_data.iloc[:, 2:-3]
    #     train_y = train_data['gauss_error']
    #     test_y = test_data['gauss_error']

    #     scaler = MinMaxScaler()
    #     train_x = scaler.fit_transform(train_x)
    #     test_x = scaler.transform(test_x)

    #     x_tensor =  torch.from_numpy(train_x).float() #torch.Size([24589, 257])
    #     y_tensor =  torch.from_numpy(train_y.values.ravel()).float() # torch.Size([24589, 1])
    #     xtest_tensor =  torch.from_numpy(test_x).float() #torch.Size([13235, 257])
    #     ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float() 
    # else:
        #Scale data
    
    #SCALING
    scaler = MinMaxScaler()
    #Scale X data
    train_x = train_data.drop(columns = ['date', 'system', 'error'], axis = 1)
    #train_x = train_data.iloc[:, 2:-1] #drop date and systems at beginning, error at the end
    train_x = scaler.fit_transform(train_x)
    test_x = test_data.drop(columns = ['date', 'system', 'error'], axis = 1)
    test_x = scaler.transform(test_x)
    #Extract prediction data (y data)
    train_y = train_data['error']
    test_y = test_data['error']
    #DATA TO TENSORS
    x_tensor =  torch.from_numpy(train_x).float() #torch.Size([24589, 257])
    y_tensor =  torch.from_numpy(train_y.values.ravel()).float() # torch.Size([24589, 1])
    y_tensor = y_tensor.unsqueeze(1) #Adds one dimension to the tensor to make them comparable
    #CREATE TRAIN DATASET
    train_ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)

    #CREATE TEST DATASET
    xtest_tensor =  torch.from_numpy(test_x).float() #torch.Size([13235, 257])
    ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float() 
    
    #For the validation/test dataset
    ytest_tensor = ytest_tensor.unsqueeze(1)
    test_ds =  torch.utils.data.TensorDataset(xtest_tensor, ytest_tensor)
    

    return train_ds, test_ds, test_y


def use_gen_data(
        mean_var_error,
        mean_var_noError
):
    """ Generates data for the models sanity check; Data has same dimensions as original data.
        Generated data is normal distributed with varying mean/var for error/non-error

        Parameters
        ----------
        the mean/variance of the error datapoints
        the mean/variance of the non-error datapoints 

        Returns
        ------
        test_df: original dataset created
        train_ds and test_ds: tensor datasets of test and train data
        test_y: original prediciton data with test train split to evaluate the model
        """
    ## Generate Data
    len_cols = 128
    len_data = 37000

    colnames_mean = [f'mean_{i + 1}' for i in range(len_cols)]
    colnames_var = [f'var_{i + 1}' for i in range(len_cols)]

    # Pre-allocate memory for the entire array
    data = np.empty((len_data, 2 * len_cols + 1))

    for i in range(len_data):
        error = np.random.choice([0, 1], p=[1 - 0.05, 0.05])
        data[i, 0] = error

        if error == 1:
            # Generate mean values
            data[i, 1:len_cols + 1] = np.random.default_rng().normal(mean_var_error[0], mean_var_error[1], size=len_cols)
            # Generate var values
            data[i, len_cols + 1:] = np.random.default_rng().normal(mean_var_error[2], mean_var_error[3], size=len_cols)
        else:
            # Generate mean values
            data[i, 1:len_cols + 1] = np.random.default_rng().normal(mean_var_noError[0], mean_var_noError[1], size=len_cols)
            # Generate var values
            data[i, len_cols + 1:] = np.random.default_rng().normal(mean_var_noError[2], mean_var_noError[3], size=len_cols)

    # Create DataFrame
    test_df = pd.DataFrame(data, columns=['error'] + colnames_mean + colnames_var)

    # Convert columns to appropriate data types
    test_df['error'] = test_df['error'].astype(int)

    split = 0.75
    train_rows = int(len(test_df) * split)
    train_data = test_df.head(train_rows)
    test_rows = int(len(test_df) * (1-split))
    test_data = test_df.tail(test_rows)

    scaler = MinMaxScaler()
    train_x = train_data.iloc[:, 1:]
    train_x = scaler.fit_transform(train_x)
    test_x = test_data.iloc[:, 1:]
    test_x = scaler.transform(test_x)
    train_y = train_data['error']
    test_y = test_data['error']

    #DATA TO TENSORS
    x_tensor =  torch.from_numpy(train_x).float() #torch.Size([24589, 257])
    y_tensor =  torch.from_numpy(train_y.values.ravel()).float() # torch.Size([24589, 1])
    y_tensor = y_tensor.unsqueeze(1) #Adds one dimension to the tensor to make them comparable
    #CREATE TRAIN DATASET
    train_ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)

    #CREATE TEST DATASET
    xtest_tensor =  torch.from_numpy(test_x).float() #torch.Size([13235, 257])
    ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float() 
    #For the validation/test dataset
    ytest_tensor = ytest_tensor.unsqueeze(1)
    test_ds =  torch.utils.data.TensorDataset(xtest_tensor, ytest_tensor)

    return test_df, train_ds, test_ds, test_y