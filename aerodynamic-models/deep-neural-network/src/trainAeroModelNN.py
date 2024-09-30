import scipy as sp
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.onnx
import torch.jit
from torch.autograd import Variable
import random as random
import time as time
from sklearn.model_selection import train_test_split
from myNNlib import paramNet


############################ SCRIPT COMMANDS ################################
TRAIN_NN_MODEL = True # True if you want to train the NN model
EXPORT_ONNX    = True # True if you want to save the NN model
LOAD_NN_MODEL  = not TRAIN_NN_MODEL # True if you want to load the NN model

############################ PATH DEFINITIONS ###############################
matFilePath = pathlib.Path(__file__).parents[1] / "dataset" / "datasetFull.mat"

# Device will determine whether to run the training on GPU or CPU.
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################## LOAD DATASET VARIABLES ###########################

# Load dataset .mat file
dataset = sp.io.loadmat(matFilePath)

# Load variables from dataset
pitchAngle = dataset['pitchAngles_full']
yawAngle = dataset['yawAngles_full']
windDirection = dataset['windDirection_full']
jointPos = dataset['jointPosDeg_full']
linkCdAs = dataset['linkCdAs_matrix']
linkClAs = dataset['linkClAs_matrix']
linkCsAs = dataset['linkCsAs_matrix']

linkAeroForces = np.concatenate((linkCdAs, linkClAs, linkCsAs), axis=1)

# Split variables for training and validation
datasetSplittingSeed = 56

pitchAngle_train, pitchAngle_val = train_test_split(pitchAngle, test_size=0.2, random_state=datasetSplittingSeed)
yawAngle_train, yawAngle_val = train_test_split(yawAngle, test_size=0.2, random_state=datasetSplittingSeed)
windDirection_train, windDirection_val = train_test_split(windDirection, test_size=0.2, random_state=datasetSplittingSeed)
jointPos_train, jointPos_val = train_test_split(jointPos, test_size=0.2, random_state=datasetSplittingSeed)

linkAeroForces_train, linkAeroForces_val = train_test_split(linkAeroForces, test_size=0.2, random_state=datasetSplittingSeed)

# from arrays to tensors
pitchAngle_train = Variable(torch.from_numpy(pitchAngle_train.transpose()).float(), requires_grad=True)
pitchAngle_val = Variable(torch.from_numpy(pitchAngle_val.transpose()).float(), requires_grad=True)
yawAngle_train = Variable(torch.from_numpy(yawAngle_train.transpose()).float(), requires_grad=True)
yawAngle_val = Variable(torch.from_numpy(yawAngle_val.transpose()).float(), requires_grad=True)
windDirection_train = Variable(torch.from_numpy(windDirection_train.transpose()).float(), requires_grad=True)
windDirection_val = Variable(torch.from_numpy(windDirection_val.transpose()).float(), requires_grad=True)
jointPos_train = Variable(torch.from_numpy(jointPos_train.transpose()).float(), requires_grad=True)
jointPos_val = Variable(torch.from_numpy(jointPos_val.transpose()).float(), requires_grad=True)

linkAeroForces_train = Variable(torch.from_numpy(linkAeroForces_train.transpose()).float(), requires_grad=True)
linkAeroForces_val  = Variable(torch.from_numpy(linkAeroForces_val.transpose()).float(), requires_grad=True)

# Move tensors to the configured device
pitchAngle_train = pitchAngle_train.to(device)  
pitchAngle_val = pitchAngle_val.to(device) 
yawAngle_train = yawAngle_train.to(device)  
yawAngle_val = yawAngle_val.to(device)  
windDirection_train = windDirection_train.to(device)
windDirection_val = windDirection_val.to(device)
jointPos_train = jointPos_train.to(device)  
jointPos_val = jointPos_val.to(device)  

linkAeroForces_train = linkAeroForces_train.to(device)  
linkAeroForces_val = linkAeroForces_val.to(device)  

###################### SOME PARAMETERS OF THE NN #########################
seed = 1000 # seed for cuda
input_parameters  = 22      # number of parameters to input layer
output_parameters = 39      # number of parameters from output layer

batch_size    = 1000        # batch size for training
learning_rate = 0.001       # learning rate for training

num_epochs     = 30000      # number of epochs for training
n_neurons      = 2**10      # neurons for each layer
num_layers     = 9          # number of hidden layers
dropout_prob   = 0.1        # dropout probability
L2_loss_weight = 0.0        # L2 loss weight

# Define the NN saving model name and path
nnModelName = "model_L" + str(num_layers) + "_N" + str(int(np.log2(n_neurons)))
if dropout_prob > 0:
    nnModelName = nnModelName + "_p" + str(dropout_prob)[2:]    
if L2_loss_weight > 0:
    nnModelName = nnModelName + "_a1" + str(np.log10(L2_loss_weight))   
nnModelName = nnModelName + "_" + str(num_epochs) + ".pt"
nnModelPath = pathlib.Path(__file__).parents[1] / "models" / nnModelName

########################### TRAINING OPERATIONS ##############################
if TRAIN_NN_MODEL:
    
    ################################ NN INIT #################################
    torch.manual_seed(seed)  # fix the seed for neural network
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

    w = torch.empty(input_parameters, n_neurons)
    nn.init.xavier_normal_(w)

    model = paramNet(input_parameters, output_parameters, num_layers, n_neurons, dropout_prob).to(device)

    print(model) # Verify the initialization of hyperparameters

    mse_cost_function = torch.nn.MSELoss()
    
    if L2_loss_weight > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_loss_weight)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Alternative cost function
    def my_loss(output, target):
        loss = torch.mean((output - target)**2)
        return loss

    ########################### TRAINING OF THE NN ###########################
    train_loss = []
    val_loss = []
    start_training = time.time()

    for epoch in range(num_epochs):

        outputs = model(windDirection_train, jointPos_train)
        loss = mse_cost_function(outputs, linkAeroForces_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train loss
        train_loss.append(loss.data.item())

        # validation loss
        val_outputs = model(windDirection_val, jointPos_val)
        val_mse = mse_cost_function(val_outputs, linkAeroForces_val)
        val_loss.append(val_mse.data.item())
        
        print('Epoch [{}/{}], train loss: {:.12f}, validation loss: {:.12f}'.format(epoch +
              1, num_epochs, loss.item(), val_mse.item()))


    time_training = time.time() - start_training
    print('time_training [min]')
    print(time_training/60)


    ######################### VISUALIZING TRAINING DATA #########################
    epochs = np.arange(num_epochs)

    plt.figure(figsize=(8, 6))
    plt.semilogy(epochs,train_loss,label="Train loss")
    plt.semilogy(epochs,val_loss,label="Validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid()
    plt.legend()
    plt.show(block=False)
    
    trainLossName = "train_loss" + nnModelName[5:-3] + ".svg"
    plt.savefig(pathlib.Path(__file__).parents[1] / "models" / trainLossName)

    ########################## SAVE THE NN MODEL ################################
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(nnModelPath) # Save the model  


############################# LOAD THE NN MODEL #################################      
else:
    # model = torch.load(nnModelPath)
    model = torch.jit.load(nnModelPath)
    model.eval()

######################### EXPORT THE MODEL TO ONNX #########################
if EXPORT_ONNX:
    batch_size = 100
    example_input = (torch.randn(3, batch_size).to(device),
                     torch.randn(19, batch_size).to(device))
    onnx_filename = nnModelName[:-3] + ".onnx"
    onnx_filepath = pathlib.Path(__file__).parents[1] / "models" / onnx_filename
    torch.onnx.export(model,               # model being run
                      example_input,                         # model input (or a tuple for multiple inputs)
                      onnx_filepath,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=13,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['windDirection', 'jointPos'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'windDirection' : {1 : 'batch_size'}, # variable length axes
                                    'jointPos' : {1 : 'batch_size'},    # variable length axes
                                    'output' : {1 : 'batch_size'}})
    print(f"Trained model saved as {onnx_filepath}")

######################### RESULT VERIFICATION #########################
linkAeroForces_predicted_train = model(windDirection_train[:,[0]], jointPos_train[:,[0]])

train_error = linkAeroForces_predicted_train - linkAeroForces_train
train_square_error = torch.pow(train_error,2)
train_mean_square_error = torch.mean(train_square_error)
print('Train verification MSE:')
print(train_mean_square_error)

###
linkAeroForces_predicted_val = model(windDirection_val, jointPos_val)

val_error = linkAeroForces_predicted_val - linkAeroForces_val
val_square_error = torch.pow(val_error,2)
val_mean_square_error = torch.mean(val_square_error)
print('Validation MSE:')
print(val_mean_square_error)

print('Model name: {}'.format(nnModelName))

# Closing all the plots
wait = input("Press Enter to close the figures.")
