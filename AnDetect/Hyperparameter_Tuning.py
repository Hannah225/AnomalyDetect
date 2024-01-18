import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Model_TestTrain
import torch
from sklearn import metrics

def dynamic_eval(test_loader, model, test_y, error_thresh = 0, use_gaussian_data = False, model_has_sigmoid = True):
  cutoff_values = [i / 100 for i in range(101)]
  evaluation = []
  model.eval()
  #Iterate over cutoff: ----------------------------------------------------------------------------------------
  for cutoff in cutoff_values:
      y_pred_list = []
      with torch.no_grad():
          for xb_test, yb_test in test_loader:
              y_test_pred = model(xb_test)
              if(model_has_sigmoid):
                out = (y_test_pred>cutoff).float() 
              else:
                out = (torch.sigmoid(y_test_pred)>cutoff).float() 
              y_pred_list.append(out)
      y_pred_list = torch.cat(y_pred_list).view(-1).tolist()
      y_true_test = test_y.values.ravel()
      if(use_gaussian_data):
        y_true_test = [1 if a_ > error_thresh else 0 for a_ in y_true_test]
      tn, fp, fn, tp = metrics.confusion_matrix(y_true_test, y_pred_list).ravel()
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = (2 * recall * precision) / (recall + precision)
      evaluation.append((cutoff, tn, fp, fn, tp, precision, recall, f1))

  #Print out best values: ----------------------------------------------------------------------------------------
  columns = ['Cutoff', 'tn', 'fp', 'fn', 'tp', 'Precision', 'Recall', 'F1']
  # Create a DataFrame from the data
  df = pd.DataFrame(evaluation, columns=columns)

  max_f1_index = df['F1'].idxmax()
  # Retrieve the corresponding row with all values
  row_with_max_f1 = df.loc[max_f1_index]

  print("Evaluation Values with the highest F1 score:")
  print("Cutoff :\t" + str(row_with_max_f1['Cutoff']))
  print("-----------")
  print("tn,fp,fn,tp")
  print(row_with_max_f1['tn'], row_with_max_f1['fp'], row_with_max_f1['fn'], row_with_max_f1['tp'])
  print("-----------")
  print("Precision:\t" + str(row_with_max_f1['Precision']))
  print("Recall:\t" + str(row_with_max_f1['Recall']))
  print("F1:\t" + str(row_with_max_f1['F1']))


  #Give out the plots: ----------------------------------------------------------------------------------------
  # Plot the change in F1 scores with respect to the cutoff values
  plt.plot(df['Cutoff'], df['F1'])
  plt.axvline(x=row_with_max_f1['Cutoff'], color='red', linestyle='--', label='Vertical Line')
  plt.xlabel("Cutoff Value")
  plt.ylabel("F1 Score")
  plt.title("Change in F1 Scores")
  plt.savefig("f1_optim")

  # Define the values you want to plot
  value_labels = {'tn': 'True Negative', 'fp': 'False Positive', 'fn': 'False Negative', 'tp': 'True Positive'}
  line = {'tn' : 12945, 'tp' : 290, 'fp' : 0, 'fn': 0}
  # Create a figure with subplots for each value
  fig, axes = plt.subplots(len(value_labels), 1, figsize=(6, 10), sharex=True)
  for i, (value, label) in enumerate(value_labels.items()):
      ax = axes[i]
      ax.plot(df['Cutoff'], df[value], label=label)
      
      # Add a horizontal solid line
      horizontal_line_y = [line[value]] * len(df)  # Adjust this line as needed
      ax.plot(df['Cutoff'], horizontal_line_y, linestyle='-', color='black')
      # Add a vertical line
      ax.axvline(x=row_with_max_f1['Cutoff'], color='red', linestyle='--')
      
      ax.set_ylabel(label)
      ax.legend()
  # Set common x-axis label
  axes[-1].set_xlabel("Cutoff Value")
  # Adjust layout and show the plots
  plt.tight_layout()
  plt.savefig("tpfp_cutoff")

  return row_with_max_f1

def dynamic_eval_noPlots(test_loader, model, test_y, model_has_sigmoid = True):
  cutoff_values = [i / 100 for i in range(101)]
  evaluation = []
  model.eval()
  #Iterate over cutoff: ----------------------------------------------------------------------------------------
  for cutoff in cutoff_values:
      y_pred_list = []
      with torch.no_grad():
          for xb_test, yb_test in test_loader:
              y_test_pred = model(xb_test)
              if(model_has_sigmoid):
                out = (y_test_pred>cutoff).float() 
              else:
                out = (torch.sigmoid(y_test_pred)>cutoff).float() 
              y_pred_list.append(out)
      y_pred_list = torch.cat(y_pred_list).view(-1).tolist()
      y_true_test = test_y.values.ravel()
      tn, fp, fn, tp = metrics.confusion_matrix(y_true_test, y_pred_list).ravel()
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = (2 * recall * precision) / (recall + precision)
      evaluation.append((cutoff, tn, fp, fn, tp, precision, recall, f1))

  #Print out best values: ----------------------------------------------------------------------------------------
  columns = ['Cutoff', 'tn', 'fp', 'fn', 'tp', 'Precision', 'Recall', 'F1']
  # Create a DataFrame from the data
  df = pd.DataFrame(evaluation, columns=columns)
  max_f1_index = df['F1'].idxmax()
  # Retrieve the corresponding row with all values
  row_with_max_f1 = df.loc[max_f1_index]
  return row_with_max_f1