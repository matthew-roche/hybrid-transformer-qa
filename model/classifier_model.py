import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Classifier model for category and possiblity prediction,

    Attributes:
        max_vocab_size (int): nn.Embedding input_size
        embed_dim (int): nn.Embedding embed_dim
    """

    def __init__(self, max_vocab_size, embed_dim, category_count, possibility_count):
        super(Classifier, self).__init__()

        self.embed = nn.Embedding(max_vocab_size, embed_dim)
        self.relu = nn.ReLU()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=2,
                dim_feedforward=2,
                dropout=0.4,
                activation='relu',
                batch_first=True
            ),
            num_layers=1
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc_category = nn.Linear(embed_dim, category_count) # categories 0-5 (6)
        self.fc_possibility = nn.Linear(embed_dim, possibility_count) # possibility, no = 0, yes = 1

    def forward(self, x, attention_mask):
        """Model forward function

        Parameters:
            x (tensor): [batch, seq]
        
        """
        # create embeddings
        x = self.embed(x)
        # attention on the syntactic patterns
        x = self.transformer(x, src_key_padding_mask=attention_mask)

        # restructure the order for pooling
        x = x.permute(0, 2, 1)

        # perform pooling, avg takes a hint of all sequences, max only picks max activated
        x_avg = self.avg_pool(x).squeeze(2)
        x_max = self.max_pool(x).squeeze(2)

        # activate non linearity
        x_max = self.relu(x_max)

        # dropout before passing to linear multi class layer
        # double edged sword, benefits for generalize, but sets 30% input to 0
        x_max = self.dropout(x_max)

        # category can pick the most activated = emphasized
        category = self.fc_category(x_max)

        # activate non linearity
        x_avg = self.relu(x_avg)
        x_avg = self.dropout(x_avg)

        # consider all sequences, because max activated may not mean most accurate
        possibility = self.fc_possibility(x_avg)

        return category, possibility