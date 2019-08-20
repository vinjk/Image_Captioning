import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        #embedding layer converting words to vectors
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        
        #embedded word vectors to hidden state output-lstm
        self.lstm_layer = nn.LSTM(embed_size, hidden_size, num_layers, bias=True, batch_first=True, dropout=0,bidirectional=False)
        
        #Linear layer
        self.linear_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch_size = features.shape[0]
        
        captions = captions[:, :-1]
        embedding = self.embed_layer(captions)
        
        # Concatenate the features and caption inputs 
        inputs = torch.cat((features.unsqueeze(1), embedding), 1)
        
        #Input to lstm
        lstm_output, _ = self.lstm_layer(inputs, None)
        
        # Convert LSTM outputs to word predictions
        output = self.linear_layer(lstm_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        stop_idx = 1
        output_cap = []
        lstm_state = None
        
        for i in range(max_len):
            lstm_output, lstm_state = self.lstm_layer(inputs, lstm_state)
            output = self.linear_layer(lstm_output)
            
            #prediction
            prediction = torch.argmax(output, dim=2)
            predicted_idx = prediction.item()
            output_cap.append(predicted_idx)
            
            #check for stop word
            if predicted_idx == stop_idx:
                break
            
            #update inputs for next
            inputs = self.embed_layer(prediction)
            
        return output_cap