import scipy as sp
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.jit
from torch.autograd import Variable
import random as random
import time as time
from sklearn.model_selection import train_test_split
import optuna

from myNNlib import paramNet

############################ PATH DEFINITIONS ###############################
matFilePath = pathlib.Path(__file__).parents[1] / "dataset" / "datasetFull.mat"

##################### FIXED PARAMETERS OF THE NN ########################
seed = 1000 # seed for cuda
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1000
input_parameters = 21
output_parameters = 39
learning_rate = 0.001 

dropout_prob = 0.0
num_epochs = 20000

##################### CUDA SEED INITIALIZATION ##########################
torch.manual_seed(seed)  # fix the seed for neural network
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed)

####################### NEURAL NETWORK FUNCTIONS ########################    

def nn_init(input_parameters, output_parameters, num_layers, n_neurons, dropout_prob, weight_decay):
    
    w = torch.empty(input_parameters, n_neurons)
    nn.init.xavier_normal_(w)
    model = paramNet(input_parameters, output_parameters, num_layers, n_neurons, dropout_prob).to(device)
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay) 
    return model, mse_cost_function, optimizer


def nn_train(model, mse_cost_function, optimizer, pitchAngle_train, yawAngle_train, jointPos_train, linkAeroForces_train):
    
    train_loss = []
    start_training = time.time()
    
    for epoch in range(num_epochs):
    
        outputs = model(pitchAngle_train, yawAngle_train, jointPos_train)
        loss = mse_cost_function(outputs, linkAeroForces_train)
    
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss.append(loss.data.item())
        final_loss = loss.data.item()
        
    time_training = time.time() - start_training
    
    return final_loss

def test_eval(model, mse_cost_function, pitchAngle_test, yawAngle_test, jointPos_test, linkAeroForces_test):
    
    outputs = model(pitchAngle_test, yawAngle_test, jointPos_test)
    test_loss = mse_cost_function(outputs, linkAeroForces_test)
    return test_loss

def dataset_loader(matFilePath):
    
    # Load dataset .mat file
    dataset = sp.io.loadmat(matFilePath)

    # Load variables from dataset
    pitchAngle = dataset['pitchAngles_full']
    yawAngle = dataset['yawAngles_full']
    jointPos = dataset['jointPosDeg_full']
    linkCdAs = dataset['linkCdAs_matrix']
    linkClAs = dataset['linkClAs_matrix']
    linkCsAs = dataset['linkCsAs_matrix']

    linkAeroForces = np.concatenate((linkCdAs, linkClAs, linkCsAs), axis=1)
    
    return pitchAngle, yawAngle, jointPos, linkAeroForces


def dataset_splitter(pitchAngle, yawAngle, jointPos, linkAeroForces):

    datasetSplittingSeed = 56

    pitchAngle_train, pitchAngle_test = train_test_split(pitchAngle, test_size=0.2, random_state=datasetSplittingSeed)
    yawAngle_train, yawAngle_test = train_test_split(yawAngle, test_size=0.2, random_state=datasetSplittingSeed)
    jointPos_train, jointPos_test = train_test_split(jointPos, test_size=0.2, random_state=datasetSplittingSeed)

    linkAeroForces_train, linkAeroForces_test = train_test_split(linkAeroForces, test_size=0.2, random_state=datasetSplittingSeed)

    return pitchAngle_train, pitchAngle_test, yawAngle_train, yawAngle_test, jointPos_train, jointPos_test, linkAeroForces_train, linkAeroForces_test


def train_prep(pitchAngle_train, yawAngle_train, jointPos_train, linkAeroForces_train):
    
    # from arrays to tensors
    pitchAngle_train = Variable(torch.from_numpy(pitchAngle_train.transpose()).float(), requires_grad=True)
    yawAngle_train   = Variable(torch.from_numpy(yawAngle_train.transpose()).float(), requires_grad=True)
    jointPos_train   = Variable(torch.from_numpy(jointPos_train.transpose()).float(), requires_grad=True)
    
    linkAeroForces_train = Variable(torch.from_numpy(linkAeroForces_train.transpose()).float(), requires_grad=True)
    
    # Move tensors to the configured device
    pitchAngle_train = pitchAngle_train.to(device)  # input
    yawAngle_train = yawAngle_train.to(device)  # input
    jointPos_train = jointPos_train.to(device)  # input
    
    linkAeroForces_train = linkAeroForces_train.to(device)  #  CFD data

    return pitchAngle_train, yawAngle_train, jointPos_train, linkAeroForces_train

def test_prep(pitchAngle_test, yawAngle_test, jointPos_test, linkAeroForces_test):
    
    # from arrays to tensors
    pitchAngle_test = Variable(torch.from_numpy(pitchAngle_test.transpose()).float(), requires_grad=True)
    yawAngle_test   = Variable(torch.from_numpy(yawAngle_test.transpose()).float(), requires_grad=True)
    jointPos_test   = Variable(torch.from_numpy(jointPos_test.transpose()).float(), requires_grad=True)
    
    linkAeroForces_test = Variable(torch.from_numpy(linkAeroForces_test.transpose()).float(), requires_grad=True)
    
    # Move tensors to the configured device
    pitchAngle_test = pitchAngle_test.to(device)  # input
    yawAngle_test = yawAngle_test.to(device)  # input
    jointPos_test = jointPos_test.to(device)  # input
    
    linkAeroForces_test = linkAeroForces_test.to(device)  #  CFD data
    
    return pitchAngle_test, yawAngle_test, jointPos_test, linkAeroForces_test


################### OBJECTIVE FUNCTION FOR OPTUNA #######################
def objective(trial):

    # 2. Suggest values of the hyperparameters using a trial object.
    # dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    num_layers  = trial.suggest_int('n_layers', 6, 10)
    neurons_exp = trial.suggest_int('neurons_exp', 7, 11)
    n_neurons = 2**neurons_exp

    model, mse_cost_function, optimizer = nn_init(input_parameters, output_parameters, num_layers, n_neurons, dropout_prob, weight_decay)
    
    pitchAngle, yawAngle, jointPos, linkAeroForces = dataset_loader(matFilePath)
    pitchAngle_train, pitchAngle_test, yawAngle_train, yawAngle_test, jointPos_train, jointPos_test, linkAeroForces_train, linkAeroForces_test = dataset_splitter(pitchAngle, yawAngle, jointPos, linkAeroForces)
    pitchAngle_train, yawAngle_train, jointPos_train, linkAeroForces_train = train_prep(pitchAngle_train, yawAngle_train, jointPos_train, linkAeroForces_train)
    pitchAngle_test,  yawAngle_test,  jointPos_test,  linkAeroForces_test  = test_prep(pitchAngle_test,  yawAngle_test,  jointPos_test,  linkAeroForces_test)
    
    train_loss = nn_train(model, mse_cost_function, optimizer, pitchAngle_train, yawAngle_train, jointPos_train, linkAeroForces_train)
    test_loss  = test_eval(model, mse_cost_function, pitchAngle_test, yawAngle_test, jointPos_test, linkAeroForces_test)
    
    return train_loss, test_loss

############################ OPTUNA ################################
study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))


############################ RESULTS ################################
fig = optuna.visualization.plot_pareto_front(study, target_names=["train loss", "test mse"])
fig.show(block=False)

print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

trial_with_lowest_train_loss = max(study.best_trials, key=lambda t: t.values[0])
print(f"Trial with lowest train loss: ")
print(f"\tnumber: {trial_with_lowest_train_loss.number}")
print(f"\tparams: {trial_with_lowest_train_loss.params}")
print(f"\tvalues: {trial_with_lowest_train_loss.values}")


fig = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="train_loss"
)
fig.show(block=False)

fig = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[1], target_name="test_loss"
)
fig.show(block=False)