from torch import nn
import torch
import torch.nn.functional as F

import numpy as np




class ConvNN(nn.Module):

    '''
    Simple convolutional NN, inspired by https://github.com/pietz/language-recognition
    inspired by https://github.com/pietz/language-recognition
    '''

    def __init__(self, n_languages = 5,
                  dropout = 0.9, kernel_size = 3, n_channels = [32, 64, 128, 256, 512, 1024], **kwargs):

        super().__init__()

        #h, w = input_width, input_heigh

        self.conv1 = nn.Conv2d(1, n_channels[0], kernel_size=kernel_size, stride=1, padding='same')
        #h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer, when padding=0
        #self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.act1 = nn.ELU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #h, w = h//2, w//2 #size reduction in the MaxPool layer

        self.conv2 = nn.Conv2d(n_channels[0], n_channels[1], kernel_size=kernel_size, stride=1, padding='same')
        #h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer, when padding=0
        #self.bn2 = nn.BatchNorm2d(n_channels[1])
        self.act2 = nn.ELU()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #h, w = h//2, w//2 #size reduction in the MaxPool layer

        self.conv3 = nn.Conv2d(n_channels[1], n_channels[2], kernel_size=kernel_size, stride=1, padding='same')
        #h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer, when padding=0
        #self.bn3 = nn.BatchNorm2d(n_channels[2])
        self.act3 = nn.ELU()
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #h, w = h//2, w//2 #size reduction in the MaxPool layer

        #self.conv4 = nn.Conv2d(n_channels[2], n_channels[3], kernel_size=kernel_size, stride=1, padding='same')
        #h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer, when padding=0
        #self.bn4 = nn.BatchNorm2d(n_channels[3])
        #self.act4 = nn.ELU()
        #self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #h, w = h//2, w//2 #size reduction in the MaxPool layer

        #self.conv5 = nn.Conv2d(n_channels[3], n_channels[4], kernel_size=kernel_size, stride=1, padding='same')
        #h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer, when padding=0
        #self.bn5 = nn.BatchNorm2d(n_channels[3])
        #self.act5 = nn.ELU()
        #self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        #h, w = h//2, w//2 #size reduction in the MaxPool layer


        self.flt = nn.Flatten()

        #self.fc6 = nn.Linear(h * w * n_channels[2], n_channels[3])
        self.fc6 = nn.LazyLinear(n_channels[3])
        self.act6 = nn.ELU()
        self.dp6 = nn.Dropout(dropout)

        self.fc7 = nn.Linear(n_channels[3], n_languages)


    def forward(self, x):

        x = torch.unsqueeze(x,1)

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.act1(out)
        out = self.mp1(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.act2(out)
        out = self.mp2(out)

        out = self.conv3(out)
        #out = self.bn3(out)
        out = self.act3(out)
        out = self.mp3(out)

        #out = self.conv4(out)
        #out = self.bn4(out)
        #out = self.act4(out)
        #out = self.mp4(out)

        #out = self.conv5(out)
        #out = self.bn5(out)
        #out = self.act5(out)
        #out = self.mp5(out)

        out = self.flt(out)

        out = self.fc6(out)
        out = self.act6(out)
        out = self.dp6(out)

        out = self.fc7(out)

        return out

class RNN(nn.Module):

    '''
    RNN from Wan, Li, et al. "Tuplemax loss for language identification." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.

    '''

    def __init__(self, device, n_languages = 5, input_heigh = 20,
                 hidden_size = [1024, 768, 512, 258], projection_size = 256,
                 N_layers = 4, **kwargs):

        super().__init__()

        self.device = device

        self.input_size = input_heigh

        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.N_layers = N_layers

        self.lstm_cells = []

        for layer_idx in range(self.N_layers):
            if layer_idx==0:
                self.lstm_cells.append(
                    nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size[layer_idx], proj_size = projection_size))
            else:
                self.lstm_cells.append(
                    nn.LSTM(input_size = self.projection_size, hidden_size = self.hidden_size[layer_idx], proj_size = projection_size))

        self.act = nn.ReLU()
        self.fc = nn.Linear(self.projection_size, n_languages)

    def forward(self, x):

        hidden_state, cell_state = [], []

        for layer_idx in range(self.N_layers):

            hidden_state.append(torch.zeros(1, x.size(0), self.projection_size).to(self.device))
            cell_state.append(torch.zeros(1, x.size(0), self.hidden_size[layer_idx]).to(self.device))

        out = torch.permute(x, (2,0,1)) #n_samples, n_batch, n_freq

        for layer_idx in range(self.N_layers):

            if layer_idx==0:
                hidden_state[layer_idx], (_, cell_state[layer_idx]) = self.lstm_cells[layer_idx](out, (hidden_state[layer_idx], cell_state[layer_idx]))
            else:
                hidden_state[layer_idx], (_, cell_state[layer_idx]) = self.lstm_cells[layer_idx](hidden_state[layer_idx-1], (hidden_state[layer_idx], cell_state[layer_idx]))

        out = self.fc(hidden_state[self.N_layers-1][-1])
        #out = self.fc(out)

        return out

    def to(self, device):

        self.device = device

        new_self = super().to(self.device)

        for layer_idx in range(self.N_layers):
            self.lstm_cells[layer_idx].to(self.device)

        return new_self


class CRNN_bartz(nn.Module):

    '''
    CRNN from Bartz, Christian, et al. "Language identification using deep convolutional recurrent neural networks." International conference on neural information processing. Springer, Cham, 2017.
    '''

    def __init__(self, device, n_languages = 5, input_height = 20,
                 lstm_layers=1, dropout=0.4, **kwargs):

        super().__init__()

        self.device = device

        self.image_height = input_height

        self.lstm_layers = lstm_layers

        self.hin = input_height//2//2*32

        self.lstm_hout = 64

        #input size: N_batch x 1 x image_height x L

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(7,7), padding='same')
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5,5), padding='same')
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding='same')
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3,3), padding='same')
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3,3), padding='same')
        self.bn5 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2,stride=2)

        self.rnn = nn.LSTM(self.hin, self.lstm_hout, self.lstm_layers, bidirectional=True, batch_first=True) #Hin, Hout

        self.flatten = nn.Flatten()

        self.fc1 = nn.LazyLinear(n_languages)

    def forward(self, x):

        x = torch.unsqueeze(x,1)

        n_batch = x.shape[0]

        h0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)
        c0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.mp(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.mp(out)

        #out = self.conv3(out)
        #out = self.relu(out)
        #out = self.bn3(out)

        #out = self.conv4(out)
        #out = self.relu(out)
        #out = self.bn4(out)

        #out = self.conv5(out)
        #out = self.relu(out)
        #out = self.bn5(out)

        out = out.view(n_batch,self.hin,-1)

        out = torch.permute(out, (0, 2, 1))

        out, (hn, cn) = self.rnn(out, (h0, c0))

        out = self.flatten(out)

        out = self.fc1(out)


        return out


class CRNN_alashban(nn.Module):

    '''
    CRNN from Alashban, Adal A., et al. "Spoken Language Identification System Using Convolutional Recurrent Neural Network." Applied Science 12.18 (2022): 9181.
    '''

    def __init__(self, device, n_languages = 5, input_height = 20,
                 lstm_layers = 1, dropout = 0.4, **kwargs):

        super().__init__()

        self.device = device

        self.image_height = input_height

        self.lstm_layers = lstm_layers

        self.hin = ((input_height-1)//10+1)*12

        self.lstm_hout = 128

        #input size: N_batch x 1 x image_height x L

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(12, 12, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(12, 12, kernel_size=5, padding='same')
        self.conv4 = nn.Conv2d(12, 12, kernel_size=5, padding='same')
        self.conv5 = nn.Conv2d(12, 12, kernel_size=5, padding='same')

        self.bn5 = nn.BatchNorm2d(12)
        self.elu = nn.ELU()

        self.average_pool = nn.AvgPool2d(kernel_size=1, stride=10)

        self.rnn = nn.LSTM(self.hin, self.lstm_hout, self.lstm_layers, batch_first=True) #Hin, Hout

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.LazyLinear(7)
        self.fc2 = nn.Linear(7, n_languages)

    def forward(self, x):

        x = torch.unsqueeze(x,1)

        n_batch = x.shape[0]

        h0 = torch.zeros(1*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)
        c0 = torch.zeros(1*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)

        out = self.conv1(x)
        out = self.conv2(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.bn5(out)
        out = self.elu(out)

        out = self.average_pool(out)

        out = out.view(n_batch,self.hin,-1)

        out = torch.permute(out, (0, 2, 1))

        out, (hn, cn) = self.rnn(out, (h0, c0))

        out = self.flatten(out)

        out = self.dropout(out)

        out = self.fc1(out)

        out = self.fc2(out)

        return out



class AttRnn(nn.Module):

    '''
    Attention RNN from
    De Andrade, Douglas Coimbra, et al. "A neural attention model for speech command recognition." arXiv preprint arXiv:1808.08929 (2018).
    '''

    def __init__(self, device, n_languages = 5, input_height = 20,
                 lstm_layers=2, dropout=0.25, **kwargs):

        super().__init__()

        self.device = device

        self.image_height = input_height

        self.lstm_layers = lstm_layers
        self.lstm_hout = 256

        #input size: N_batch x 1 x image_height x L

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 1, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn2 = nn.BatchNorm2d(1)

        #convert dimensions to N_batch x L x image_height here!

        self.rnn = nn.LSTM(self.image_height, self.lstm_hout, self.lstm_layers, bidirectional=True, batch_first=True) #Hin, Hout

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_languages)

    def forward(self, x):

        n_batch, n_features, L = x.shape

        x = torch.unsqueeze(x,1)

        h0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)
        c0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.permute(out, (0, 3, 2, 1)).squeeze() #remove last singleton dimension

        out, (hn, cn) = self.rnn(out, (h0, c0))

        ################################Attention Block################################

        xFirst = out[:,-1,:] #first element in the sequence

        query = self.fc1(xFirst)

        #dot product attention
        attScores = torch.bmm(query.unsqueeze(1), out.permute((0,2,1))) / np.sqrt(2*self.lstm_hout)

        attScores = self.softmax(attScores)

        attVector = torch.bmm(attScores, out).squeeze()

        ##############################################################################

        out = self.fc2(attVector)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc4(out)

        return out


#model = AttRnn().to(device)

#input = torch.randn(3, 40, 10).to(device) #N_batch x 1 x image_height x L

#model(input).shape


class CRNN(nn.Module):

    def __init__(self, device, n_languages = 5, input_height = 20,
                 lstm_layers=2, dropout=0.25, **kwargs
                ):

        '''
        Convolutional RNN with the same number of parameters as above attention RNN
        '''

        super().__init__()

        self.device = device

        self.image_height = input_height

        self.lstm_layers = lstm_layers
        self.lstm_hout = 128

        #input size: N_batch x 1 x image_height x L

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 1, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn2 = nn.BatchNorm2d(1)

        #convert dimensions to N_batch x L x image_height here!

        self.rnn = nn.LSTM(self.image_height, self.lstm_hout, self.lstm_layers, bidirectional=True, batch_first=True) #Hin, Hout

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.LazyLinear(32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, n_languages)

    def forward(self, x):

        x = torch.unsqueeze(x,1)

        n_batch = x.shape[0]

        h0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)
        c0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.permute(out, (0, 3, 2, 1)).squeeze() #remove last singleton dimension

        out, (hn, cn) = self.rnn(out, (h0, c0))

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc4(out)

        return out


#model = crnn(image_height=20).to(device)

#input = torch.randn(3, 20, 10).to(device) #N_batch x 1 x image_height x L

#model(input).shape


from .tdnn import TDNN

class X_vector(nn.Module):
    """
    Created on Sat May 30 19:59:45 2020
    large
    @author: krishna
    repo: https://github.com/KrishnaDN/x-vector-pytorch
    """
    def __init__(self, n_languages = 5, input_height = 20, dropout=0., **kwargs):
        super().__init__()
        self.tdnn1 = TDNN(input_dim=input_height, output_dim=512, context_size=5, dilation=1,dropout_p=dropout)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2,dropout_p=dropout)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3,dropout_p=dropout)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=dropout)
        self.tdnn5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1,dropout_p=dropout)
        #### Frame levelPooling
        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, n_languages)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        inputs = torch.permute(inputs, (0, 2, 1))
        tdnn1_out = self.tdnn1(inputs)
        #return tdnn1_out
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        return predictions

class X_vector_small(nn.Module):
    """
    male
    Created on Sat May 30 19:59:45 2020
    @author: krishna
    repo: https://github.com/KrishnaDN/x-vector-pytorch
    """
    def __init__(self, n_languages = 5, input_height = 20, **kwargs):
        super().__init__()
        self.tdnn1 = TDNN(input_dim=input_height, output_dim=512, context_size=5, dilation=1,dropout_p=0.1)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.1)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.1)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.1)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.1)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, n_languages)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        inputs = torch.permute(inputs, (0, 2, 1))
        tdnn1_out = self.tdnn1(inputs)
        #return tdnn1_out
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        return predictions

class AttRnn_small(nn.Module):

    '''
    Attention RNN from
    De Andrade, Douglas Coimbra, et al. "A neural attention model for speech command recognition." arXiv preprint arXiv:1808.08929 (2018).
    '''

    def __init__(self, device, n_languages = 5, input_height = 20,
                 lstm_layers=2, dropout=0.25, **kwargs):

        super().__init__()

        self.device = device

        self.image_height = input_height

        self.lstm_layers = lstm_layers
        self.lstm_hout = 64

        #input size: N_batch x 1 x image_height x L

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 1, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn2 = nn.BatchNorm2d(1)

        #convert dimensions to N_batch x L x image_height here!

        self.rnn = nn.LSTM(self.image_height, self.lstm_hout, self.lstm_layers, bidirectional=True, batch_first=True) #Hin, Hout

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_languages)

    def forward(self, x):

        n_batch, n_features, L = x.shape

        x = torch.unsqueeze(x,1)

        h0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)
        c0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.permute(out, (0, 3, 2, 1)).squeeze() #remove last singleton dimension

        out, (hn, cn) = self.rnn(out, (h0, c0))

        ################################Attention Block################################

        xFirst = out[:,-1,:] #first element in the sequence

        query = self.fc1(xFirst)

        #dot product attention
        attScores = torch.bmm(query.unsqueeze(1), out.permute((0,2,1))) / np.sqrt(2*self.lstm_hout)

        attScores = self.softmax(attScores)

        attVector = torch.bmm(attScores, out).squeeze()

        ##############################################################################

        out = self.fc2(attVector)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc4(out)

        return out


class AttRnn_exp(nn.Module):

    '''
    Attention RNN from
    De Andrade, Douglas Coimbra, et al. "A neural attention model for speech command recognition." arXiv preprint arXiv:1808.08929 (2018).
    '''

    def __init__(self, device, n_languages = 5, input_height = 20,
                 lstm_layers=3, dropout=0.25, **kwargs):

        super().__init__()

        self.device = device

        self.image_height = input_height

        self.lstm_layers = lstm_layers
        self.lstm_hout = 128

        #input size: N_batch x 1 x image_height x L

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 1, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn2 = nn.BatchNorm2d(1)

        #convert dimensions to N_batch x L x image_height here!

        self.rnn = nn.LSTM(self.image_height, self.lstm_hout, self.lstm_layers, bidirectional=True, batch_first=True) #Hin, Hout

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_languages)

    def forward(self, x):

        n_batch, n_features, L = x.shape

        x = torch.unsqueeze(x,1)

        h0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)
        c0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.permute(out, (0, 3, 2, 1)).squeeze() #remove last singleton dimension

        out, (hn, cn) = self.rnn(out, (h0, c0))

        ################################Attention Block################################

        xFirst = out[:,-1,:] #first element in the sequence

        query = self.fc1(xFirst)

        #dot product attention
        attScores = torch.bmm(query.unsqueeze(1), out.permute((0,2,1))) / np.sqrt(2*self.lstm_hout)

        attScores = self.softmax(attScores)

        attVector = torch.bmm(attScores, out).squeeze()

        ##############################################################################

        out = self.fc2(attVector)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc4(out)

        return out
