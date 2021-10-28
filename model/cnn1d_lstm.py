import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN1D_LSTM(nn.Module):
    def __init__(self, args, device):
        super(CNN1D_LSTM_V1, self).__init__()      
        self.args = args

        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_data_channel = args.num_channel
        self.sincnet_bandnum = args.sincnet_bandnum
        
        feature_extractor = args.enc_model
        self.feat_models = nn.ModuleDict([
            ['psd', PSD_FEATURE()],
            ['sincnet', SINCNET_FEATURE(args=args,
                        num_eeg_channel=self.num_data_channel) # padding to 0 or (kernel_size-1)//2
                        ]])
        self.feat_model = self.feat_models[feature_extractor]

        if args.enc_model == "psd":
            self.feature_num = 7
        elif args.enc_model == "sincnet":
            self.feature_num = args.cnn_channel_sizes[args.sincnet_layer_num-1]

        self.conv1dconcat_len = self.feature_num * self.num_data_channel
        
        activation = 'relu'
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['relu', nn.ReLU(inplace=True)],
            ['tanh', nn.Tanh()],
            ['sigmoid', nn.Sigmoid()],
            ['leaky_relu', nn.LeakyReLU(0.2)],
            ['elu', nn.ELU()]
        ])

        # Create a new variable for the hidden state, necessary to calculate the gradients
        self.hidden = ((torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, args.batch_size, self.hidden_dim).to(device)))

        def conv1d_bn(inp, oup, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv1d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(oup),
                self.activations[activation],
                nn.Dropout(self.dropout),
        )

        self.features = nn.Sequential(
            conv1d_bn(self.conv1dconcat_len,  256, 21, 1, 10),
            nn.MaxPool1d(kernel_size=4, stride=4),
            conv1d_bn(256, 512, 5, 1, 2),
            conv1d_bn(512, 512, 5, 1, 2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )        

        # self.agvpool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=self.hidden_dim,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout) 
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features= 256, bias=True),
            nn.BatchNorm1d(256),
            self.activations[activation],
            nn.Linear(in_features=256, out_features= args.output_dim, bias=True),
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feat_model(x)
        # print("1x: ", x.shape)
        x = torch.reshape(x, (self.args.batch_size, self.conv1dconcat_len, x.size(3)))
        # print("2x: ", x.shape)
        x = self.features(x)
        # print("3x: ", x.shape)
        x = x.permute(0, 2, 1)
        self.hidden = tuple(([Variable(var.data) for var in self.hidden]))
        output, self.hidden = self.lstm(x, self.hidden)    
        output = output[:,-1,:]
        logit = self.classifier(output)
        # proba = nn.functional.softmax(output, dim=1)
        # exit(1)
        return logit, self.hidden

    def init_state(self, device):
        self.hidden = ((torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device), torch.zeros(self.num_layers, self.args.batch_size, self.hidden_dim).to(device)))
     
    # def init_state(self, batch_size, device):
    #     return (torch.zeros(self.num_layers, batch_size, 1024).to(device),
    #         torch.zeros(self.num_layers, batch_size, 1024).to(device))
        # hs_forward = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        # cs_forward = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

        # return (torch.nn.init.kaiming_normal_(hs_forward), 
        #     torch.nn.init.kaiming_normal_(cs_forward))
