import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, cohen_kappa_score

import Process_data
from Model_TestTrain import MLP_Residual, train_model, test_model
from Hyperparameter_Tuning import dynamic_eval


#put function name in a container:
data_sources = [Process_data.get_normal_data, 
                Process_data.get_scaled_robust_data, 
                Process_data.get_scaled_standard_data,
                Process_data.get_pca_data,
                Process_data.get_umap_data,
                #Process_data.get_merk_data,
                Process_data.get_gauss_data
                ]

data_names = ["Normal", "Scaled.Robust", "Scaled.Standard", "PCA", "UMAP", 
              #"Merk",
            "Gauss"]
counter_datasources = 0 #for the outer for loop; i used in another loop

number_of_splits = 12

for fn in data_sources:

    print(data_names[counter_datasources])

    #reintialize resulzts lists
    result_dataframes = []
    cohen_dataframes = [] 

    for x in range(number_of_splits):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Available Device: {device}")

        ##GET THE DATA
        gen_data = False #do we want to use generated data for the sanity check
        use_gaussian_data = False 

        #get the original data and save the data name
        data_name = data_names[counter_datasources]
        #use a different function for data preprocessing each time
        seed = x+50
        train_df, test_df, train_x, test_x, train_y, test_y = fn(seed = seed)

        train_ds, test_ds = Process_data.mlp_data(train_x, test_x, train_y, test_y)

        ##BOILERPLATE
        n_input_dim = train_ds.__getitem__(1)[0].shape[0] #256 Variables each; we access the first tensor in our ds and then get the shape of the tensor
        n_hidden1 = 1024
        n_hidden2 = 512
        n_hidden3 = 128
        n_hidden4 = 8
        n_output = 1
        #number of blocks for the resiudal model:
        n_blocks = 4

        batchsize = 512
        learning_rate = 0.0001
        epochs = 1


        error_thresh = 0.5 #Later used to evaluate model (cutoff)

        #INITIALIZE DATALOADER AND LOSS FUNCTION

        #Currently not used but another option to weigh error samples
            # test_weights = [15 if x == 1 else 1 for x in train_y]
            # sampler = WeightedRandomSampler(
            #     weights=test_weights,
            #     num_samples=batchsize,
            #     replacement=False
            # )
            

        #Weighted Loss Function
        class_weight = torch.FloatTensor([15])
        class_weight = class_weight.to(device)
        loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        loss_func.to(device)
            #loss_func = nn.BCEWithLogitsLoss() #We can still apply this even with gaussian filter because we have values between 0 and 1
            #loss_func = nn.BCELoss()

        #Dataloader
        train_loader = DataLoader(train_ds, batch_size=batchsize,
                                #generator=torch.Generator(device=device),
                                #sampler=sampler
                                shuffle = True, 
                                #pin_memory=True
        )
        test_loader = DataLoader(test_ds, batch_size=batchsize
                                #, generator=torch.Generator(device=device)
                                )

        ##SET UP MODEL
        
        model = MLP_Residual(n_input_dim, n_hidden1, n_hidden3, n_output, n_blocks)
        model.to(device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps = 0.00001, weight_decay= 0.01)

        train_loss = train_model(train_loader, test_loader, model, optimizer, loss_func, epochs, device=device)
        print('Last iteration loss value: ' + str(train_loss[-1]))
        plt.figure()
        plt.plot(train_loss, '-', color='orange', label = "Training Loss")
        #plt.plot(test_loss, '-', color='blue', label = "Testing Loss", alpha = 0.6)
        plt.legend(loc='best')
        name_loss = data_name + '_loss.png'
        plt.savefig(name_loss)

        mlp_pred = test_model(error_thresh, test_loader, model, test_y, device, use_gaussian_data = use_gaussian_data,
                model_has_sigmoid= False
                )


        #===================================  OTHER CLASSIFIERS ==================================================================

        # Define classifiers
        n_neighbors = 7
        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        #clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
        #svc = SVC()
        logistic_model = LogisticRegression(solver = 'newton-cholesky', max_iter = 1000)
        isolation_forest = IsolationForest(n_estimators=200)
        #Fitting
        knn_classifier.fit(train_x, train_y)
        #clf.fit(train_x)
        isolation_forest.fit(train_x)
        #svc.fit(train_x, train_y)
        logistic_model.fit(train_x, train_y)

        # Predict using the trained classifiers
        knn_predictions = knn_classifier.predict(test_x)
        #clf_pred = clf.fit_predict(test_x)
        #clf_pred = np.array([False if i == 1 else True for i in clf_pred])
        i_f_p = isolation_forest.predict(test_x)
        isolation_forest_predictions = (i_f_p == 1)
        #svc_predictions = svc.predict(test_x)
        logistic_predicitions = logistic_model.predict(test_x)


        anomaly_score_predictions, likelihood_predictions = Process_data.get_AnomalyPred(train_df, test_df)

        # Generate random predictions based on the error rate and test dataset size
        error_rate = np.mean(test_y)
        random_predictions = np.random.choice([True, False], size=len(test_y), p=[error_rate, 1 - error_rate])

        # dynamic_eval(test_loader, model, test_y, 
        #             use_gaussian_data = use_gaussian_data,
        #             model_has_sigmoid= False,
        #             error_thresh = error_thresh
        #         )


        #======================== OBTAIN THE RESULTS ============================================================
        #gemeinsamer dataframe erzeugen:
        df = pd.DataFrame({'True_val': test_y,
                        'Random': random_predictions, 
                        'Logistic': logistic_predicitions,
                        #'SVC': svc_predictions,
                        'IsolationForest': isolation_forest_predictions,
                        #'CLF': clf_pred,
                        'KNN': knn_predictions,
                        'MLP': mlp_pred,
                        'Anomaly': anomaly_score_predictions,
                        'Likelihood': likelihood_predictions
                        })

        # results of: f1, precision und recall von jeder predicition
        f1 = []
        recall = []
        precision = []

        for column in df.columns:
            f1.append(f1_score(test_y, df[column]))
            recall.append(recall_score(test_y, df[column]))
            precision.append(precision_score(test_y, df[column]))

        df_results = pd.DataFrame({
            'Prediction' : df.columns,
            'F1' : f1,
            'Precision': precision,
            'Recall': recall
        })

        result_dataframes.append(df_results)


        #cohens kappa matrix
        cohen_matrix = [[0 for x in range(len(df.columns))] for y in range(len(df.columns))] 
        i = 0
        for column1 in df.columns:
            j = 0
            for column2 in df.columns:
                cohen_matrix[i][j] = cohen_kappa_score(df[column1], df[column2])
                j+=1
            i+=1


        df_cohen = pd.DataFrame(
            cohen_matrix, columns=df.columns,
        )
        df_cohen.insert(0, 'Method', df.columns)

        cohen_dataframes.append(df_cohen)


    
    # GET THE OVERALL RESULTS FOR EACH RUN   
    mean_results = pd.DataFrame()

    # Iterate over each dataframe and calculate the mean for each cell
    for df in result_dataframes:
        if mean_results.empty:
            mean_results = df.copy()  # Copy the structure of the first dataframe
        else:
            mean_results[1:] += df[1:]  # Add the values of the current dataframe to the mean dataframe

    # for column in mean_results.columns[1:]:
    #     mean_results[column] = mean_results[column].astype(int)

    # Divide each cell in the mean dataframe by the number of dataframes to get the mean
    for column in mean_results.columns[1:]:
        for index in mean_results.index:
            mean_results.at[index, column] /= len(result_dataframes)

    mean_cohen = pd.DataFrame()

    # Iterate over each dataframe and calculate the mean for each cell
    for df in cohen_dataframes:
        if mean_cohen.empty:
            mean_cohen = df.copy()  # Copy the structure of the first dataframe
        else:
            mean_cohen[1:] += df[1:]  # Add the values of the current dataframe to the mean dataframe

    # for column in mean_cohen.columns[1:]:
    #     mean_cohen[column] = mean_cohen[column].astype(int)

    # Divide each cell in the mean dataframe by the number of dataframes to get the mean
    for column in mean_cohen.columns[1:]:
        for index in mean_cohen.index:
            mean_cohen.at[index, column] /= len(cohen_dataframes)
    
    name_results = 'results/'+ data_name + '_Results.csv'
    name_cohen = 'results/'+ data_name + '_Cohen.csv'

    mean_results.to_csv(name_results, sep=',', index=False, encoding='utf-8')
    mean_cohen.to_csv(name_cohen, sep=',', index=False, encoding='utf-8')

    print("Results obtained.")

    counter_datasources += 1