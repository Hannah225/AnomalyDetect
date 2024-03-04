import torch
from sklearn import metrics
from torch import nn
from tqdm import tqdm


class MLPResBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPResBlock, self).__init__()
        
        self.layers = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, input_size),
          nn.ReLU(),
          nn.LayerNorm(input_size)
        )

    def forward(self, x):
        out = self.layers(x)
        return out + x

class MLP_Residual(nn.Module):
    def __init__(self, input_size, hidden_size, second_hidden, output_size, num_blocks):
        super(MLP_Residual, self).__init__()

        self.first_layer = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.ReLU()
        )

        self.blocks = nn.ModuleList([MLPResBlock(hidden_size, second_hidden) for _ in range(num_blocks)])

        self.final_layer = nn.Sequential(
          nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out = self.first_layer(x)

        for block in self.blocks:
            out = block(out)
        
        out = self.final_layer(out)
        return out
    

def train_model(train_dl, test_loader, model, optimizer, loss_func, epochs, device, model_has_sigmoid = False):
  train_loss = []
  test_loss = []
  loop = tqdm(range(epochs), leave=False, total=epochs)
  for epoch in loop:
    loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
    model.train()
    for xb, yb in train_dl:
      xb = xb.to(device)
      yb = yb.to(device)
      #print("xb yb Device", xb.device, yb.device)
      optimizer.zero_grad()
      y_pred = model(xb)
      #print("pred Device", y_pred.device)
      loss = loss_func(y_pred, yb)
      #print("loss Device", loss.device)
      loss.backward()
      optimizer.step()

    model.eval()
    with torch.no_grad():
      for xb_test, yb_test in test_loader:
        xb_test = xb_test.to(device)
        yb_test = yb_test.to(device)
        y_test_pred = model(xb_test)
        valid_loss = loss_func(y_test_pred, yb_test)
    
    loop.set_postfix(loss=loss.item())
    train_loss.append(loss.item())
    test_loss.append(valid_loss.item())
  return train_loss, test_loss



def test_model(error_thresh, test_loader, model, test_y, device, use_gaussian_data = False, model_has_sigmoid = True):
  y_pred_list = []
  model.eval()
  with torch.no_grad():
    for xb_test, yb_test in test_loader:
      xb_test = xb_test.to(device)
      y_test_pred = model(xb_test)
      if(model_has_sigmoid):
        y_pred_tag = torch.round(y_test_pred)
      else: #Add sigmoid manually when evaluating!
        y_pred_tag = torch.round(torch.sigmoid(y_test_pred))
      y_pred_list.append(y_pred_tag)
  y_pred_list = torch.cat(y_pred_list).view(-1).tolist()
  y_true_test = test_y.values.ravel()
  if(use_gaussian_data):
    y_true_test = [1 if a_ > error_thresh else 0 for a_ in y_true_test]
  print("Number of Errors with this error_thresh:\t" + str(sum(y_true_test)))
  a = metrics.confusion_matrix(y_true_test, y_pred_list).ravel()
  print("Confusion Matrix of the Test Set")
  print("-----------")
  print("tn,fp,fn,tp")
  print(a)
  print("-----------")
  print("Precision of the MLP :\t" + str(metrics.precision_score(y_true_test, y_pred_list)))
  print("Recall of the MLP    :\t" + str(metrics.recall_score(y_true_test, y_pred_list)))
  print("F1 Score of the Model :\t" + str(metrics.f1_score(y_true_test, y_pred_list)))
  return y_pred_list