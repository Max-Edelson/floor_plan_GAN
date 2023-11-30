import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, latent_dim, max_sequence_length):
        super().__init__()
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(embedding_dim + latent_dim, 256),
            # nn.LSTMCell(256, 256),
            nn.LSTMCell(256, 256)
        ])
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(256, vocab_size)
        )
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.msl = max_sequence_length

    def forward(self, x):

        batch_size = x.size(0)
        hidden_states = [torch.zeros(batch_size, 256).to(device) for _ in range(3)]
        cell_states = [torch.zeros(batch_size, 256).to(device) for _ in range(3)]
        outputs = []
        input = torch.cat((x[:, 0, :], torch.zeros((x.shape[0], self.embedding_dim)).to(device)), dim=1)
        for i in range(self.msl): # TODO largest sequence length
            for j in range(2): # layers of LSTM
                hidden_states[j], cell_states[j] = self.lstm_cells[j](input)
                input = self.drop(hidden_states[j])

            # batch size x Vocab size
            output = self.fc(hidden_states[-1])
            outputs.append(F.softmax(output, dim=1))
            if i < self.msl - 1: # max length - 1

                output_tokens = sample(output)
                output_embeddings = self.embedding(output_tokens).squeeze(1)

                # x and output embeddings will have different shapes! 
                input = torch.cat((x[:, i+1, :], output_embeddings), dim=1)
        

        return torch.stack(outputs, dim=1)
    
def sample(logits, temperature=1.0):

    # Apply temperature
    logits = logits / temperature

    # Draw a sample from the distribution
    return torch.multinomial(F.softmax(logits, dim=1), 1)
    

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 32, num_layers=2, dropout=0.18, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(32 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        pdb.set_trace()

        out = self.embedding(x.long())
        out, _ = self.lstm(out)
        out = out[:, -1, :]
        return self.fc(out)
    