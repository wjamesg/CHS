# DL_XAI_1.py:  Synthesizes code that runs DL and shap
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_squared_error, r2_score)
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import torch.nn.init as init
import optuna
import statsmodels.api as sm   # statsmodels package for regression
from captum.attr import IntegratedGradients
import shap

# Some basic input parameters
binary_outcome  = False  # True or False (capitalized)
use_study       = False # Include study indicators?  If so, 2-component network will be used (simpleNN_2)
PATHWAY_AS_INTERMEDIATE = False
datadir         = "C:/jim/Sab/Sab1/NNPlusXAI/"
filename        = "simulationChallenge2.csv"  # CRC_deltad includes both original D and study adjusted deltad
outFileBase     = "simChallenge2"
codeCodom       = False  # additively coded g in filename will be recoded into two codom indicators in split_columns
Path2Pathway_File = "pathways2.csv"
testSize        = 0.1  # what fraction of full dataset allocated to testing?
valSize         = 0.1  # what fraction of full dataset to allocate to validation
n_epochs_trial  = 50  # how many epochs during optuna trial phase?
n_epochs_test   = 200  # how many epochs during final training/testing phase?
batchsize_test  = 512   # how many per batch for test evaluation (ignored for all training)
ranNum          = 123  # Initial seed value for random number generator
tune            = False # True: use optuna to train;  False: train using fixed set of hyperparms
prune           = False # if tuning, do you want to prune epochs that do not appear to be improvements?
# t_: Tuning set of hyperparameters to be considered by Optuna, if tune=True
t_n_trials          = 100 # how many trials to run
tLR_lo              = 0.001  # learning rate  values will be log sampled on this range
tLR_hi              = 0.1
tDrop_lo            = 0.05   # dropout  values sampled on this range
tDrop_hi            = 0.4
t_activation_name   = ['relu', 'leaky_relu', 'elu']
t_weight_init       = ['None']  # 'kaiming_uniform'
tBatch_lo           = 16      # sampled on this range
tBatch_hi           = 64
t_weight_decay      = [0, 1e-4, 1e-3]     # this should be numbers representing categories to try
# structure of the main network and if use_study, the second network
t_nHidden_lo        = 1       # sampled
t_nHidden_hi        = 3
t_nNeurons_lo       = 8       # sampled
t_nNeurons_hi       = 512
t_n_hidden2         = 0   # how many hidden layers in second (study+) network (zero will use study as bias parms)
t_n_neurons2        = 8   # how many neurons per hidden layer in first network (irrelevant if hidden2=0)
# Note: the n_out_neurons1 value is set below and is not currently tunable

# weight decay is penalty on # parms of form weight_decay*sum(weights^2) so model does not get too complex

# f_: Fixed values for running training/val only once, if tune=False (ignored if doing optuna)
# first set up the two parts of the network
f_dropout_rate        = 0.25
f_activation_name     = "Linear" # no activation...should approximate logistic or linear regression
f_activation_name     = "Softplus" # activation fct for hidden layers in first network
f_batch_size          = 512
f_weight_decay        = 0.0  # >0 does regularization...default to zero
f_lr_init             = 0.001 # initial lr used by scheduler
f_lr_step_size        = 20    # lr stepsize used by scheduler
f_lr_gamma            = 1.0   # fraction to cut lr every f_lr_step_size used by scheduler (set to 1.0 to not cut)
f_n_hidden            = 2     # how many hidden layers in first network
f_n_neurons           = 16    # how many neurons per hidden layer in first network
# ***********************************************************************
n_out_neurons1        = 64    # how many 'outputs' from first network...not tunable at present but could be added
                              # ...can be thought of as an additional hidden layer in the combined network
f_n_hidden2            = 0    # how many hidden layers in second network
f_n_neurons2           = 8    # how many neurons per hidden layer in second network if n_hidden2 >= 1
f_activation_name2     = "Softplus" # activation fct for hidden layers in second network if n_hidden2 >= 1

PATHWAY_ARRAY_LENGTH = 1  # The length of the subarray which represent one pathway node
NUMBER_OF_COVARIATES = 16 # The number of covariates for pathway as intermediate model and they are the columns at the left side of the input file

# set device to gpu if it exists or to cpu otherwise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# function to read data from above input file

def readData(dir,file):
    # ************** Data input and preprocessing **************
    # set data up for linear  regression and NN
    # First read in all the data from the csv file
    # read data into dataframe...'infer' option reads variable names from first line
    df = pd.read_csv(f"{dir}{file}")

    # Print the resulting dataframe
    print(f"head: {df.head()}")
    print(f"tail: {df.tail()}")
    print(f"shape: {df.shape}") #  Full data has 63,253 subjects which agrees w/ SAS
    print(f"describe: {df.describe()}")
    print(f"columns: {df.columns}")
    print(f"dtypes: {df.dtypes}")

    # use df.columns to get list of variable names...then block copy into list below
    return df

# function to input dataframe and return X,y numpy arrays and X,y torch tensors. Used by get_Xten_yten function
def dfToTensor(df, xcols, ycol):
    global y_sd  # Jim added this
    global y_L1_d
    # Create tensors X and y from dataframe
    print("xcols")
    print(xcols)
    X = df[xcols].values  # numpy arrays
    y = df[ycol].values
    y_sd = np.std(y)  # Jim added this
    y_L1_d = np.mean(np.abs(y-np.mean(y)))
    X = torch.tensor(X, dtype=torch.float32)  # tensors
    # Reshape y to be a column vector. -1 says figure out how many rows; 1 says make it 1 column
    # Mingzhi, why do we need to use the view for y but not X?
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y

# function to generate tensors from df
def get_Xten_yten(indf):
    # declare numbers of x and y elements to be global
    global n_x,n_x2,n_x3,n_y
    global XAI_list

    # indf = pd.get_dummies(indf, columns=['agecat'], dtype=int)

    # pick one for target outcome
    ycol  = ["y"]






    # choose which variables to include in this run (ctrl O/ to comment a block of code)
    xcols = ['e', 'z', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10']

    n_x  = len(xcols)
    n_y  = len(ycol)
    print(f"numy        : {n_y}")
    print(f"numX        : {n_x}")
    print(f"X           : {xcols}")

    # create tensors from input df and desired x and y variables
    Xten, yten = dfToTensor(indf, xcols, ycol)
    # get counts of y
    # following converts tensor to a numpy array with dimensions N,1 (where N is number of subjects)
    yarr = yten.cpu().numpy()
    # following converts N, 1 array to a vector of N elements
    yarr = yarr.ravel()
    # following needs vector of N elements as input
    print(f"Counts of y : {pd.Series(yarr).value_counts()}")
    # create dummy tensor used only to apply .shape[1] to get firstnumX for use below
    print(f"Mean of y   : {np.mean(yten.detach().cpu().numpy())}")

    return Xten, yten, xcols

# function to split X and y into train, validation, and test sets
def trainValTest(Xten, yten, testsize, valsize):
    # Read full-data X and y and split into train/val/test
    # first, split data into temp (train/val) and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        Xten, yten, test_size=testsize, random_state=ranNum)

    # split temp into train and val
    # compute vsize so that resulting validation sets are of the desired input size valsize
    trainsize = 1 - testsize
    vsize = valsize / trainsize
    print(f"Proportion of training data allocated to validation={vsize}")
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=vsize, random_state=ranNum)
    # how many observations and X,y variables in the above datasets
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_val.shape  : {X_val.shape}")
    print(f"y_val.shape  : {y_val.shape}")
    print(f"X_test.shape : {X_test.shape}")
    print(f"y_test.shape : {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def getPerformanceBinary(method,actuals,predictions,probabilities):
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    auc = roc_auc_score(actuals, probabilities)
    # compute Tjur R-squared...there is probably a better way to do this
    # we need the mean of the probabilities for D=1 and for D=0
    mean_1 = 0.0
    mean_0 = 0.0
    n_1 = 0.0
    n_0 = 0.0
    length = len(actuals)
    for i in range(length):
        if actuals[i] == 1:
            mean_1 += probabilities[i]
            n_1 += 1
        elif actuals[i] == 0:
            mean_0 += probabilities[i]
            n_0 += 1
    mean_1 = mean_1 / n_1
    mean_0 = mean_0 / n_0
    tjur_rsq = abs(mean_1 - mean_0)

    print(f"sum of actuals: {sum(actuals)}")
    print(f"sum of preds  : {sum(predictions)}")
    print(f"Accuracy    : {accuracy}")
    print(f"Precision   : {precision}")
    print(f"Recall      : {recall}")
    print(f"F1 Score    : {f1}")
    print(f"AUC Score   : {auc}")
    print(f"Tjur Rsq    : {tjur_rsq}")
    # plot histogram of predicted values
    plt.plot(range(0,1))
    plt.ylabel("Frequency")
    plt.xlabel(f"Predicted probabilities by {method}")
    plt.hist(probabilities)
    plt.show()
    result = pd.DataFrame([[method,auc,f1,tjur_rsq]],columns=['Method','AUC','f1','tjurRsq'])
    return result

def getPerformanceContinuous(method,actuals,predictions):
    mse = mean_squared_error(actuals, predictions)
    rsq = round(r2_score(actuals, predictions),4)
    # rsq = r2_score(actuals, predictions)
    print(f"Rsq: {rsq}")
    print(f"MSE: {mse}")

    # generate scatterplot of predicted vs actuals with best-fit line between them
    # scatterplot of predicted vs. actual Y
    # fit regression line...straight line has degree 1
    b, a = np.polyfit(actuals.ravel(), predictions.ravel(), deg=1)
    # Create sequence of 10 numbers from min to max of actuals (could actually get by with only 2 numbers...min,max)
    min_x=min(actuals)
    print(f"min={min}")
    max_x=max(actuals)
    print(f"max={max}")
    xseq = np.linspace(min_x, max_x, num=10)
    # Plot regression line over range of X
    plt.plot(xseq, a + b * xseq, color='red', lw=2.5)
    plt.scatter(actuals, predictions, color='blue')
    plt.title(f"{method}: R-squared = {rsq}")
    plt.ylabel("Predictions")
    plt.xlabel(f"Actuals")
    plt.show()
    result = pd.DataFrame([[method,rsq,mse]],columns=['Method','Rsq','MSE'])
    return result

# functions to run standard linear and logistic regression
def runLinear(Xfit, yfit, Xtest, ytest, label):
    print(f"*** Linear Regression on {label} ***")

    model = LinearRegression()
    model.fit(Xfit, yfit)

    print("  Intercept:", model.intercept_)
    print("  Coefficients:", model.coef_)

    y_pred = model.predict(Xtest)  # get predictions
    perfLR = getPerformanceContinuous("LR",ytest,y_pred)
    # perf.to_csv("testperf.csv")
    return model, perfLR

def runLogistic(Xfit,yfit,Xtest,ytest,label):
    Xfit,yfit,Xtest,ytest = Xfit.cpu().numpy(), yfit.cpu().numpy(), Xtest.cpu().numpy(), ytest.cpu().numpy() # convert tensor to numpy array as statsmodel can't work with tensors
    print(f"*** Logistic Regression on {label} ***")

    yfit = yfit.ravel()
    model = LogisticRegression()
    model.fit(Xfit, yfit)

    print("  Intercept:", model.intercept_)
    print("  Coefficients:", model.coef_)
    y_pred_proba = model.predict(Xtest) # get probability prediction on test set
    actual = sum(ytest==1)
    threshold = 1 - (actual / len(ytest)) # use threshold or test_threshold?  needs to agree with NN performance
    y_pred = (y_pred_proba >= test_threshold).astype(int) # get 0/1 prediction, using global threshold
    perfLR = getPerformanceBinary("LR",ytest,y_pred,y_pred_proba)
    return model, perfLR

# required by Pytorch
class CustomDataset(Dataset):
    # 'features' below are the X values; 'labels' are the y values
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# *** Read Data ***
indf = readData(datadir,filename)

def split_columns(indf, cols):
    for col in cols:
        if col in indf.columns:
            indf[f'{col}_1'] = indf[col].apply(lambda x: 1 if x == 1 else 0)
            indf[f'{col}_2'] = indf[col].apply(lambda x: 1 if x == 2 else 0)
            indf.drop(columns=[col], inplace=True)
    return indf

# Jim added block below to optionally recode g using Mingzhi's split_columns function
if codeCodom:
    # Jim this is configured for our 10-snp simulation...needs to be generalized if using other setups
    col2change = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10']
    # Following written by Mingzhi to split additive g into codominant before setting xcol
    indf = split_columns(indf, col2change)
    # following re-set for use in setting xcol in get_Xten_yten and for shap interaction code
    gene_list = ['g1_1', 'g1_2', 'g2_1', 'g2_2', 'g3_1', 'g3_2', 'g4_1',
                 'g4_2', 'g5_1', 'g5_2', 'g6_1', 'g6_2', 'g7_1', 'g7_2', 'g8_1', 'g8_2',
                 'g9_1', 'g9_2', 'g10_1', 'g10_2']

# abstract desired X and y variables for this analysis and create X and y tensors
Xten,yten,xcols = get_Xten_yten(indf)

# create training, validation, and testing tensors from X and y datasets
X_train, X_val, X_test, y_train, y_val, y_test = trainValTest(Xten, yten, testSize, valSize)

# put data on device
X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

# threshold = (sum(yten == 1)/len(yten)).item() # set threshold= '# of 1/sample size' for prediction, used for both NN and logistic reg
# following is proportino of cases in whole sample.  we actually want the proportion in the sample being evaluated
# for accuracy etc.
test_threshold = (sum(yten == 1)/len(yten)).item() # set threshold= '# of 1/sample size' for prediction, used for both NN and logistic reg
test_threshold = 1 - (sum(y_test == 1)/len(y_test)).item()  # set threshold= '# of 1/sample size' for prediction, used for both NN and logistic reg
test_threshold = 0.5  # Jim which one to use?
print(f"test threshold={test_threshold}")

# ************** End Data input and preprocessing **************
def getSHAP_Main(model_type,model,X, y_sd, xcols, baseline):
    print(f"computing shap Main for {model_type}")
    # convert tensor to numpy array
    if model_type=='LR':
        X = X.cpu().numpy()  # input X is tensor...convert to array
        # get number of subjects in X
        baseline_numpy = baseline.detach().cpu().numpy()
        masker = shap.maskers.Independent(baseline_numpy)

        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(X)
        LR_shapL1 = np.mean(np.abs(shap_values), axis=0) / y_sd
        LR_shapL2 = np.sqrt(np.mean(shap_values * shap_values, axis=0)) / y_sd

        # Convert to DataFrame for better readability
        list_of_tuples = list(zip(xcols, LR_shapL1, LR_shapL2))
        result = pd.DataFrame(list_of_tuples, columns=['Feature', 'LR_ShapL1', 'LR_ShapL2'])
    elif model_type=='NN':
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        explainer = shap.DeepExplainer(model, baseline)

        # Initialize an array to COLLECT feature importance
        shap_values = []

        # Calculate SHAP values for each batch in the test loader
        for batch_features, _ in X:
            batch_features = batch_features.to(device)

            # Calculate SHAP values for the batch
            shap_batch = explainer.shap_values(batch_features, check_additivity=False)
            shap_batch = shap_batch.squeeze(-1)
            shap_values.append(shap_batch)  # Jim added squaring of shap_value here

        # Convert list of SHAP values to numpy array and average across all samples
        shap_values = np.concatenate(shap_values, axis=0)

        NN_shapL1 = np.mean(np.abs(shap_values), axis=0) / y_sd
        NN_shapL2 = np.sqrt(np.mean(shap_values * shap_values, axis=0)) / y_sd  # jim added this

        # Convert to DataFrame for better readability
        list_of_tuples = list(zip(xcols, NN_shapL1, NN_shapL2))
        result = pd.DataFrame(list_of_tuples, columns=['Feature', 'NN_ShapL1', 'NN_ShapL2'])

    print(f"done with shap Main for {model_type}")
    return result

def get_shap_interaction_adjusted(i, j, model, model_type, norm_type, y_sd, data_loader, baseline, device):  # baseline should be a tensor
    # a=True
    # Initialize the SHAP DeepExplainer with the model and background data
    if model_type=="LR":         # baseline should be a tensor
        baseline_numpy = baseline.detach().cpu().numpy()
        masker = shap.maskers.Independent(baseline_numpy)
        explainer = shap.LinearExplainer(model, masker)
        baseline = baseline[:].to(device)

    elif model_type=="NN":           # baseline should be a tensor
        explainer = shap.DeepExplainer(model, baseline)
        baseline = baseline[:].to(device)

    interaction_values = []
    for batch_features, _ in data_loader:
        batch_features = batch_features.to(device)
        batch_size = batch_features.size(0)

        # Ensure the batch size matches the baseline
        current_baseline = baseline[:batch_size]

        # SHAP values with feature i set to baseline
        features_i_baseline = batch_features.clone()
        features_i_baseline[:, i] = current_baseline[:, i]

        # SHAP values with feature j set to baseline
        features_j_baseline = batch_features.clone()
        features_j_baseline[:, j] = current_baseline[:, j]

        if model_type == "LR":
            shap_values_full = explainer.shap_values(batch_features)
            shap_values_i_baseline = explainer.shap_values(features_i_baseline)
            shap_values_j_baseline = explainer.shap_values(features_j_baseline)

        elif model_type == "NN":
            shap_values_full = explainer.shap_values(batch_features, check_additivity=False)
            shap_values_i_baseline = explainer.shap_values(features_i_baseline, check_additivity=False)
            shap_values_j_baseline = explainer.shap_values(features_j_baseline, check_additivity=False)

            shap_values_full=shap_values_full.squeeze(-1)
            shap_values_i_baseline = shap_values_i_baseline.squeeze(-1)
            shap_values_j_baseline = shap_values_j_baseline.squeeze(-1)

        interaction_value = (
                    shap_values_full[:,i] + shap_values_full[:,j]  - shap_values_i_baseline[:,j] - shap_values_j_baseline[:,i])
        interaction_values.append(interaction_value)

    # Concatenate and compute the mean interaction values
    interaction_values = np.concatenate(interaction_values, axis=0)
    if norm_type == 'L1':
        mean_interaction_values = np.mean(np.abs(interaction_values), axis=0) / y_sd
    if norm_type == 'L2':
        mean_interaction_values = np.sqrt(np.mean(interaction_values * interaction_values, axis=0)) / y_sd
    return mean_interaction_values

# function to get shap interaction results based on Mingzhi's latest code
def getSHAP_Intxn(model_type, norm_type, model,X,Xlist1,Xlist2,y_sd,baseline, dic, device):
    # Note: Data sent to Xlist1 and Xlist2 should be shapIntxnV1 and shapIntxnV2 defined above

    print(f"computing shap Intxn for {model_type}")
    interaction_matrix = pd.DataFrame(index=Xlist1, columns=Xlist2)
    label = []
    label.append(model_type)
    dflabel = pd.DataFrame(label)

    for i in range(len(Xlist1)):
        for j in range(len(Xlist2)):
            interaction_shap = get_shap_interaction_adjusted(dic[Xlist1[i]], dic[Xlist2[j]], model, model_type, norm_type, y_sd, X, baseline, device)
            interaction_matrix.loc[Xlist1[i], Xlist2[j]] = interaction_shap
    # interaction_matrix.to_csv(output_csv_path, index=True)
    result = pd.concat([dflabel, interaction_matrix])
    print(f"done with shap Intxn for {model_type}")
    return result

def get_sample(data_loader, sample_size):
    all_features = []
    for batch_features, batch_labels in data_loader:
        all_features.append(batch_features)

    all_features = torch.cat(all_features, dim=0)

    if sample_size > all_features.size(0):
        raise ValueError("Sample size exceeds the total number of rows in combined_loader!")
    indices = torch.randperm(all_features.size(0))[:sample_size]
    sampled_features = all_features[indices]
    return sampled_features

# ************** Run Standard Regression  **************
# Run 1: Fit/eval on full/full data to cross-check with SAS
# Run 2: Fit/eval on train/test data for comparison with NN
if binary_outcome:
    modelFull, perf_LRfull   = runLogistic(Xten,yten,Xten,yten,'full data')
    modelTrain, perf_LRtest  = runLogistic(X_train,y_train,X_test,y_test,'train/test')
else:
    modelFull, perf_LRfull  = runLinear(Xten,yten,Xten,yten,'full data')
    modelTrain, perf_LRtest = runLinear(X_train,y_train,X_test,y_test,'train/test')

from torch.utils.data import ConcatDataset, DataLoader

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batchsize_test, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchsize_test, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batchsize_test, shuffle=False)

combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

combined_loader = DataLoader(combined_dataset, batch_size=batchsize_test, shuffle=False)
baseline = get_sample(combined_loader, 1000)

shapMain_LR = getSHAP_Main('LR',modelTrain,X_train, y_sd, xcols, baseline)

# ************** End: Standard Regression  **************

# ************** Run Neural Network Analysis **************
print("*** Neural Network ***")
# *** Define classes that will be used in NN ***

class ActivationFunction(nn.Module):
    def __init__(self, activation_name, **kwargs):
        super().__init__()
        # Define a dictionary of supported activation functions with configurable parameters
        self.activation_funcs = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(**kwargs),  # Accepts negative_slope via kwargs
            'elu': nn.ELU(**kwargs),  # Accepts alpha via kwargs
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'Softplus': nn.Softplus(),
            'Linear': nn.Identity()
        }

        # Check if the provided activation function name is supported
        if activation_name not in self.activation_funcs:
            raise ValueError(f"Unsupported activation function: {activation_name}")

        # Set the chosen activation function
        self.activation = self.activation_funcs[activation_name]

    def forward(self, x):
        return self.activation(x)


# set up single network
class SimpleNN(nn.Module):
    def __init__(self, dropout_rate=0.19, n_hidden=2, n_neurons=64, activation_function="tanh",
                 **activation_kwargs):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(n_x, n_neurons), ActivationFunction(activation_function, **activation_kwargs)]

        # the _ in next statement represents a 'junk' variable...use when we don't want to access loop index
        # modified loop over Mingzhi's code to subtract 1...otherwise we get n_hidden+1 layers
        # range(n) begins index at 0 and goes to n-1...if n_hidden=3 below cycles through 0, and 1 (so adds 2 layers)
        for _ in range(n_hidden-1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(ActivationFunction(activation_function, **activation_kwargs))
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(n_neurons, 1))
        # No change here, but note the final activation (sigmoid or none) in forward()

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        if binary_outcome:
            return torch.sigmoid(x)  # Apply sigmoid at the output
        else:
            return x # Apply linear (none) at the output

def readPathwayLists(path):
    df_p=pd.read_csv(path)
    grouped_df = df_p.groupby('p3')['snp'].apply(list).reset_index()
    grouped_list=list(grouped_df["snp"].values)
    return grouped_list

grouped_list_indices = []
if PATHWAY_AS_INTERMEDIATE:
    grouped_list = readPathwayLists(Path2Pathway_File)
    col_name_to_index = {name: idx for idx, name in enumerate(xcols)}
    # Convert column names in grouped_list to column indices
    grouped_list_indices = [[col_name_to_index[col] for col in sublist] for sublist in grouped_list]


# ******** Step 1: Use optuna to optimize NN over hyper parameters  **************
# Dictionary that maps the activation function names to actual PyTorch functions
activations = {
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'elu': F.elu,
    'softplus': F.softplus,
}
def objective(trial):
    # Hyperparameters to be tuned by Optuna
    dropout_rate = trial.suggest_float('dropout_rate', tDrop_lo, tDrop_hi)
    activation_name = trial.suggest_categorical('activation_function', t_activation_name)
    # weight_init = trial.suggest_categorical('weight_init', t_weight_init)
    batch_size = trial.suggest_int('batch_size', tBatch_lo, tBatch_hi)
    weight_decay = trial.suggest_categorical('weight_decay', t_weight_decay)
    lr_init = trial.suggest_float('lr_init', tLR_lo, tLR_hi, log=True)
    # apply activation dictionary to name to get function
    activation_function = activations[activation_name]

    # set number of hidden layers and nodes...this is same whether using study or not
    n_hidden = trial.suggest_int('n_hidden', t_nHidden_lo, t_nHidden_hi)
    n_neurons = trial.suggest_int('n_neurons', t_nNeurons_lo, t_nNeurons_hi)


    model = SimpleNN(dropout_rate=dropout_rate, n_hidden=n_hidden, n_neurons=n_neurons,
                             activation_function=activation_function)


    model.to(device)
    # if(weight_init == 'kaiming_uniform'):
    #     model.apply(kaiming_init)

    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)

    if binary_outcome:
        criterion = nn.BCELoss()
        # BCEWithLogitsLoss(reduction='mean'):  combines a Sigmoid layer and the BCELoss in one single class.
    else:
        criterion = nn.MSELoss()

    train_dataset   = CustomDataset(X_train, y_train)
    val_dataset     = CustomDataset(X_val, y_val)
    train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_trial_val_loss = float('inf')
    num_epochs = n_epochs_trial
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_labels = batch_labels.to(device)
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_trial_val_loss:
            best_trial_val_loss = val_loss  # Update the best validation loss if the current one is lower

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} .. Train Loss: {train_loss} .. Test Loss: {val_loss} ")
        if epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs} .. Train Loss: {train_loss} .. Test Loss: {val_loss} ")

        # should we end this trial because this hyper parm combo looks hopeless?
        if prune:
           trial.report(val_loss, epoch)
           if trial.should_prune():
                print(f"Pruning trial at epoch {epoch+1}...val_loss = {val_loss}...best_loss={best_trial_val_loss}")
                raise optuna.exceptions.TrialPruned()

    return best_trial_val_loss  # Use optuna to optimize this

def applyToTestData(model,test_loader):
    print("*** NN Performance ***")
    predictions = []
    actuals = []
    if binary_outcome:
        with torch.no_grad():
            probabilities = []
            for batch_features, batch_labels in test_loader:
                batch_labels = batch_labels.to(device)
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                probas = outputs.view(-1).tolist()  # These are continuous probabilities
                binary_preds = [1 if prob >= test_threshold else 0 for prob in probas]  # Convert to binary predictions
                predictions.extend(binary_preds)
                actuals.extend(batch_labels.view(-1).tolist())
                probabilities.extend(probas)
        perfNN = getPerformanceBinary("NN",actuals,predictions,probabilities)
    else:
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_labels = batch_labels.to(device)
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                predictions.extend(outputs)
                # actuals.extend(batch_labels.view(-1).tolist())
                actuals.extend(batch_labels)
        # need to translate actual and predictions lists into 1D vectors
        perfNN = getPerformanceContinuous("NN",np.array(actuals),np.array(predictions))
    return perfNN

# decide whether to do full training or just run a single training based on fixed hyperparms
if tune:
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(),direction='minimize')
    # set ntrials to 1/7 of total possible combinations of hyperparms
    study.optimize(objective, n_trials=t_n_trials)

    # **** Tuning is done ******
    # copy best hyperparms from optimization experiment into variables for final training
    dropout_rate = study.best_params['dropout_rate']
    n_neurons = study.best_params['n_neurons']
    n_hidden = study.best_params['n_hidden']
    activation_function = activations[study.best_params['activation_function']]
    batch_size = study.best_params['batch_size']
    weight_decay = study.best_params['weight_decay']
    lr_init = study.best_params['lr_init']
    n_hidden2            = f_n_hidden2
    n_neurons2           = f_n_neurons2
    activation_function2 = f_activation_name2
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best parameters: {study.best_params}")
    print(f"activation_function = {activation_function}")
else:
    n_hidden             = f_n_hidden
    n_neurons            = f_n_neurons
    n_hidden2            = f_n_hidden2
    n_neurons2           = f_n_neurons2
    activation_function  = f_activation_name
    activation_function2 = f_activation_name2
    dropout_rate         = f_dropout_rate
    batch_size           = f_batch_size
    weight_decay         = f_weight_decay
    lr_init              = f_lr_init

# Re-train the model based on optimal hyperparm settings determined above
# Best parameters: {'dropout_rate': 0.1, 'n_neurons': 64, 'n_hidden': 3,
# 'activation_function': 'leaky_relu', 'batch_size': 32, 'weight_decay': 0}

print("Retrain the model based on best hyperparms")

model = SimpleNN(dropout_rate=dropout_rate, n_hidden=n_hidden, n_neurons=n_neurons,
                 activation_function=activation_function)

model.to(device)
num_epochs = n_epochs_test
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=f_lr_step_size, gamma=f_lr_gamma)
criterion = nn.MSELoss()

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

best_trial_val_loss = float('inf')
best_epoch = 0
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_labels = batch_labels.to(device)
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    scheduler.step()  # update learning rate if epoch is a multiple of scheduler step_size

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_labels = batch_labels.to(device)
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    if val_loss < best_trial_val_loss:
        best_trial_val_loss = val_loss  # Update the best validation loss if the current one is lower
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        best_epoch = epoch  # store which epoch gave the best model

    # JIM...add code to store epoch values for plotting
    if epoch % 10 == 0:
        print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        print(f"Epoch {epoch + 1}/{num_epochs} .. Train Loss: {train_loss} .. Test Loss: {val_loss} ")
    if epoch == num_epochs - 1:
        print('Epoch-{0} lr: {1}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        print(f"Epoch {epoch + 1}/{num_epochs} .. Train Loss: {train_loss} .. Test Loss: {val_loss} ")

print("Training done")
plt.plot(range(num_epochs), train_losses, label='Training Loss', color='blue')
plt.plot(range(num_epochs), val_losses, label='Validation Loss', color='red')
plt.ylabel("Loss/Error")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# apply model to test data and get accuracy statistics
print("Load best model and compute performance on test data")
print(f"Best model came from epoch: {best_epoch}")

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batchsize_test, shuffle=False)

print("Apply the model to the test data")
# Question:  Do I need to re-declare model since I'm reading model from the external file?
model = SimpleNN(dropout_rate=dropout_rate, n_hidden=n_hidden, n_neurons=n_neurons,
                activation_function=activation_function)
model.to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# get performance on test data
perf_NNtest = applyToTestData(model, test_loader)

# write performance results
perf_test = pd.concat([perf_LRtest, perf_NNtest], axis=0)  # concatenate rows
perf_test.to_csv(f"{outFileBase}_ModelPerformance_Test.csv", index=False)

# get shap for main effects
shapMain_NN = getSHAP_Main('NN',model,train_loader, y_sd, xcols, baseline)
# get shapIntxn_NN for interaction effects (Mingzhi)

# write shap results to csv output file
# shapMain_LR.to_csv(f"{outFileBase}_ShapMain_LR.csv")
# shapMain_NN.to_csv(f"{outFileBase}_ShapMain_NN.csv")
# concatenate LR and NN shap Main results and write to same file
shapMain = pd.concat([shapMain_LR, shapMain_NN], axis=1)  # concatenate columns
shapMain.to_csv(f"{outFileBase}_SHAPMain_Train.csv", index=False)

dic={} #DICTIONARY for feature name to feature number
for i in range(0,len(xcols)):
    dic[xcols[i]]=i

# get_shap_interaction_adjusted(0, 1, model, "NN", "L2", y_sd, test_loader, baseline,device)

shapIntxn_LR = getSHAP_Intxn("LR", "L2", modelTrain,test_loader,['e', 'z'],['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10'],y_sd,baseline, dic, device)
shapIntxn_NN = getSHAP_Intxn("NN", "L2", model,test_loader,['e', 'z'],['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10'],y_sd,baseline, dic, device)

# write shap results to csv output file
# shapIntxn_LR.to_csv(f"{outFileBase}_ShapIntxn_LR.csv")
# shapIntxn_NN.to_csv(f"{outFileBase}_ShapIntxn_NN.csv")
# concatenate LR and NN shap Main results and write to same file
shapIntxn = pd.concat([shapIntxn_LR, shapIntxn_NN], axis=0)  # concatenate rows
shapIntxn.to_csv(f"{outFileBase}_SHAPIntxn_test.csv", index=True)

