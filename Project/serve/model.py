import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis. The model has been changed 
    to impove performances.
    
    """ 

    def __init__(self, embedding_dim, hidden_dim, vocab_size,n_layers=1, 
                 bidirectional=False, dropout=0.0):
        """
        Initialize the model by settingg up the various layers.
        
        Args:
        embedding_dim: The number of expected embedded features in input.
        hidden_dim: The number of features in the hidden state.
        vocab_size: Size of the vocabulary used for the embedding.
        n_layers: Number of recurrent layers.
        bidirectional: If True, becomes a bidirectional LSTM.
        dropout: Dropout value applyed on the outputs of each LSTM layer except the last layer.
        
        """
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if bidirectional:
            self.lstm = nn.LSTM(embedding_dim, 
                            int(hidden_dim/2), 
                            n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout)
        else :
            self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.dropout = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        #x_packed =nn.utils.rnn.pack_padded_sequence(embeds, lengths)
        lstm_out, _ = self.lstm(embeds)
        #x_unpacked = nn.utils.rnn.pad_packed_sequence(x)
        out = self.dense(self.dropout(lstm_out))
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())