import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import data
from Model_TestTrain import MLP_Residual, train_model, test_model

##GET THE DATA
gen_data = False #do we want to use generated data for the sanity check
use_gaussian_data = False #Gaussian data currently not useable (minor fixes needed)


if gen_data:
    test_df, train_ds, xtest_tensor, ytest_tensor, test_y, train_y, test_x, train_x = data.use_gen_data(
        mean_var_error = (1, 0.5, 1.5, 1), #(mu, sigma of mean dist; mu, sigma of var dist if there is an error)
        mean_var_noError = (1, 1, 1, 0.5) #(mu, sigma of mean dist; mu, sigma of var dist if there is NO error)
    )
else:
    df = data.get_data()
    train_ds, test_ds, test_y = data.data_preprocess(df)


##BOILERPLATE
n_input_dim = train_x.shape[1]
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 128
n_hidden4 = 8
n_output = 1
#number of blocks for the resiudal model:
n_blocks = 4

batchsize = 512
learning_rate = 0.001
epochs = 20

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
loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    #loss_func = nn.BCEWithLogitsLoss() #We can still apply this even with gaussian filter because we have values between 0 and 1
    #loss_func = nn.BCELoss()

#Dataloader
train_loader = DataLoader(train_ds, batch_size=batchsize,
                        #sampler=sampler
                        shuffle = True, 
                        #pin_memory=True
)
test_loader = DataLoader(test_ds, batch_size=batchsize)


##SET UP MODEL
model = MLP_Residual(n_input_dim, n_hidden1, n_hidden3, n_output, n_blocks)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps = 0.00001, weight_decay= 0.01)

train_loss, test_loss = train_model(train_loader, test_loader, model, optimizer, loss_func, epochs)
print('Last iteration loss+ value: ' + str(train_loss[-1]))

plt.plot(train_loss, '-')
plt.plot(test_loss, '-', alpha = 0.6)
plt.savefig("loss")

test_model(error_thresh, test_loader, model, test_y, use_gaussian_data = use_gaussian_data,
        model_has_sigmoid= False
        )