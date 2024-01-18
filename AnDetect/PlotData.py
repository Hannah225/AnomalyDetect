import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import stats

import Process_data


df = Process_data.get_data()

colnames_mean = df.columns[1:] #get first row 
# Calculate mean and variance per column based on another column (system)
grouped_mean_mean = df.groupby('system')[colnames_mean].agg('mean') #groupby system dann über die einzelnen systeme den aggregierten mean berechenen

colnames_var = df.columns[1:]
grouped_mean_var= df.groupby('system')[colnames_var].agg('mean')

# Sorting the system names by the mean value of mean or variance
colnames_mean_sorted = df.columns[df[colnames_mean].mean().argsort()]
colnames_var_sorted = df.columns[df[colnames_var].mean().argsort()]

#find out how many faults each system has
systems_faults = df.iloc[:, [1, -1]] #select system and error
grouped = systems_faults.groupby(['system', 'error']).size().unstack(fill_value=0) 
#get value counts of true and false values per systems, unstack to make a df out it; fill value to make nan to 0 
#(and for some reason float to int...)

# Plot the results
ax = grouped.plot(kind='bar', stacked=True, width=0.9) 
ax.set_xlabel('System')
ax.set_ylabel('Count')
ax.set_title('Count of True and False values per System')
plt.savefig("Error_Counts_Per_System")


# Group the data by system and create one plot per system
fig, ax = plt.subplots(nrows=20, ncols=3, figsize=(15, 50))
plt.subplots_adjust(hspace=1)

for i in range(grouped_mean_mean.shape[0]):
    ax_i = ax[i // 3, i % 3]

    sorted_mean = grouped_mean_mean.drop(columns = 'system').iloc[i].sort_values()
    indexes = ['var' + col[4:] if col.startswith('mean_') else col for col in sorted_mean.index]
    sorted_var = grouped_mean_var.drop(columns = 'system')[indexes].iloc[i]

    sorted = pd.concat([sorted_mean.reset_index(drop=True), sorted_var.reset_index(drop=True)], axis = 1)

    #Variance has a LOT of heavy outliers
    z_scores = np.abs(stats.zscore(sorted))
    #Define threshold for outliers
    threshold = 3
    #Remove outliers
    sorted = sorted[(z_scores < threshold).all(axis=1)]

    #Try normalization with MinMaxScalar
    #scaler = MinMaxScaler()
    #sorted = scaler.fit_transform(sorted)

    sorted.plot(ax=ax_i, legend=False)

    # chart formatting
    ax_i.legend(["Mean", "Variance"], fontsize="7", loc="upper right")
    ax_i.set_title(f'System {int(grouped_mean_mean.iloc[i][0])}', fontsize="10" )
    ax_i.set_xlabel('Parameter Columns' , fontsize="8")
    ax_i.set_ylabel('Values', fontsize="8")

plt.savefig("Systems_MeanVar_Sorted")



