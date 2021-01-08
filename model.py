import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


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
        super(DecoderRNN, self).__init__()
        self.embedding_word = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        

    def forward(self, features, captions):
        embeds_word = self.embedding_word(captions[:,: -1])
        embeds = torch.cat((features.unsqueeze(1),embeds_word),1)
        output, _ = self.lstm(embeds)
        output = self.linear(output)
        return output
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_lst = []
        for i in range(max_len):
            if i == 0:
                print(inputs.shape)
                output,(h_, c_) = self.lstm(inputs,states)
                _, predicted = torch.max(self.linear(output),dim = -1)
                predicted = predicted.squeeze(1)
                output_lst.append(predicted)
            else:
                word_emd = self.embedding_word(predicted)
                word_emd = word_emd.unsqueeze(1)
                output, (h_, c_) = self.lstm(word_emd, (h_, c_) )
                _, predicted = torch.max(self.linear(output),dim = -1)
                predicted = predicted.squeeze(1)
                output_lst.append(predicted)
        output_lst = torch.stack(output_lst,1)
        return output_lst.squeeze().tolist()