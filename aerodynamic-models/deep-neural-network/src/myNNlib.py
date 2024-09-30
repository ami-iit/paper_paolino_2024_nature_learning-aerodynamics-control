import torch
import torch.nn as nn
import torch.nn.functional as F

### Parametric Neural Network class ###
class paramNet(nn.Module):
    def __init__(self, input_parameters, output_parameters, num_layers, n_neurons, dropout_prob):
        super(paramNet, self).__init__()
        self.input_layer = nn.Linear(input_parameters,n_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_neurons, n_neurons),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob)
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(n_neurons,output_parameters)
        
    def forward(self, windDirection, jointPos):
        # Normalize inputs (degrees to [-1,1])
        # pitchAngle = torch.div(pitchAngle,180)
        # yawAngle = torch.div(yawAngle,180)
        jointPos = torch.div(jointPos,180)
        # Concatenate inputs
        input = torch.cat([windDirection, jointPos],dim=0)
        # Compute output with forward pass
        input_layer_out = F.relu(self.input_layer(input.T))
        for hidden_layer in self.hidden_layers:
            input_layer_out = hidden_layer(input_layer_out)
        output = self.output_layer(input_layer_out)
        output = torch.transpose(output,0,1)
        return output