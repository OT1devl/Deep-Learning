import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import spacy

class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
    def __len__(self):
        return len(self.itos)

class Field:
    def __init__(self, tokenize, init_token="<sos>", eos_token="<eos>", lower=True, pad_token="<pad>"):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.lower = lower
        self.vocab = None  # built with build_vocab
    
    def build_vocab(self, sentences, min_freq=1):
        freq = {}
        for sentence in sentences:
            if self.lower:
                sentence = sentence.lower()
            for tok in self.tokenize(sentence):
                freq[tok] = freq.get(tok, 0) + 1
        tokens = [tok for tok, count in freq.items() if count >= min_freq]
        tokens = sorted(tokens, key=lambda x: (-freq[x], x))
        stoi = {self.pad_token: 0, self.init_token: 1, self.eos_token: 2}
        itos = [self.pad_token, self.init_token, self.eos_token]
        for tok in tokens:
            if tok not in stoi:
                stoi[tok] = len(itos)
                itos.append(tok)
        self.vocab = Vocab(stoi, itos)

    def numericalize(self, sentence):
        if self.lower:
            sentence = sentence.lower()
        tokens = self.tokenize(sentence)
        tokens = [self.init_token] + tokens + [self.eos_token]
        return [self.vocab.stoi.get(tok, self.vocab.stoi[self.pad_token]) for tok in tokens]

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def load_csv_data(filepath):
    src_sentences, trg_sentences = [], []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_sentences.append(row["english"])
            trg_sentences.append(row["german"])
    return src_sentences, trg_sentences


class TranslationDataset:
    def __init__(self, src_sentences, trg_sentences, src_field, trg_field):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_field = src_field
        self.trg_field = trg_field
    def __len__(self):
        return len(self.src_sentences)
    def __getitem__(self, idx):
        src_num = self.src_field.numericalize(self.src_sentences[idx])
        trg_num = self.trg_field.numericalize(self.trg_sentences[idx])
        return {"src": src_num, "trg": trg_num}

class Batch:
    def __init__(self, src, trg):
        self.src = src  # (batch, src_seq_length)
        self.trg = trg  # (batch, trg_seq_length)

def pad_sequences(sequences, pad_value):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value]*(max_len - len(seq)) for seq in sequences]
    return torch.LongTensor(padded)  # batch-first

def create_batches(dataset, batch_size, pad_idx, shuffle=True):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i+batch_size]
        src_batch, trg_batch = [], []
        for idx in batch_indices:
            sample = dataset[idx]
            src_batch.append(sample["src"])
            trg_batch.append(sample["trg"])
        src_padded = pad_sequences(src_batch, pad_idx)
        trg_padded = pad_sequences(trg_batch, pad_idx)
        yield Batch(src_padded, trg_padded)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"
        self.values = nn.Linear(embed_size, embed_size)
        self.keys   = nn.Linear(embed_size, embed_size)
        self.queries= nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values  = self.values(values)    # (N, value_len, embed_size)
        keys    = self.keys(keys)          # (N, key_len, embed_size)
        queries = self.queries(query)      # (N, query_len, embed_size)
        # Divide en heads
        values  = values.reshape(N, value_len, self.heads, self.head_dim)
        keys    = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        # Producto escalar: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512,
                 num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cpu", max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def make_src_mask(self, src):
        # src: (N, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N,1,1,src_len)
        return src_mask.to(self.device)
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

source_field = Field(tokenize=tokenize_eng, init_token="<sos>", eos_token="<eos>", lower=True)
target_field = Field(tokenize=tokenize_ger, init_token="<sos>", eos_token="<eos>", lower=True)

train_src, train_trg = load_csv_data(r"")
test_src, test_trg   = load_csv_data(r"")

train_dataset = TranslationDataset(train_src, train_trg, source_field, target_field)
test_dataset  = TranslationDataset(test_src, test_trg, source_field, target_field)

source_field.build_vocab(train_src)
target_field.build_vocab(train_trg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_vocab_size = len(source_field.vocab)
trg_vocab_size = len(target_field.vocab)
src_pad_idx = source_field.vocab.stoi[source_field.pad_token]
trg_pad_idx = target_field.vocab.stoi[target_field.pad_token]

model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512,
                    num_layers=3, forward_expansion=4, heads=8, dropout=0.1, device=device, max_length=100).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

def train_model(dataset, model, optimizer, criterion, batch_size, pad_idx, epochs):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        for batch in create_batches(dataset, batch_size, pad_idx, shuffle=True):
            src = batch.src.to(device)  # (batch, src_len)
            trg = batch.trg.to(device)  # (batch, trg_len)
            # During the training, we use the complete target:
            # Input al decoder: trg[:, :-1]
            # Labels: trg[:, 1:]
            output = model(src, trg[:, :-1])
            # output: (batch, trg_len-1, trg_vocab_size)
            output = output.reshape(-1, trg_vocab_size)
            target = trg[:, 1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            print(f"Loss: {loss.item():.4f}")

def evaluate_accuracy(dataset, model, batch_size, pad_idx):
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for batch in create_batches(dataset, batch_size, pad_idx, shuffle=False):
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output = model(src, trg[:, :-1])
            predictions = output.argmax(2)  # (batch, trg_len-1)
            for i in range(predictions.shape[0]):
                for j in range(predictions.shape[1]):
                    if trg[i, j+1] != pad_idx:
                        total_tokens += 1
                        if predictions[i, j] == trg[i, j+1]:
                            correct_tokens += 1
    return correct_tokens / total_tokens if total_tokens > 0 else 0

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    tokens = src_field.tokenize(sentence.lower())
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    numericalized = [src_field.vocab.stoi.get(tok, src_field.vocab.stoi[src_field.pad_token]) for tok in tokens]
    src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(device)  # (1, src_len)
    src_mask = model.make_src_mask(src_tensor)
    enc_src = model.encoder(src_tensor, src_mask)
    
    outputs = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0).to(device)  # (1, len(outputs))
        trg_mask = model.make_trg_mask(trg_tensor)
        out = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
        # out: (1, seq_len, trg_vocab_size)
        best_guess = out[0, -1].argmax().item()
        outputs.append(best_guess)
        if best_guess == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    translated_tokens = [trg_field.vocab.itos[idx] for idx in outputs]
    return " ".join(translated_tokens[1:-1])

if __name__ == "__main__":
    epochs = 10
    batch_size = 32
    train_model(train_dataset, model, optimizer, criterion, batch_size, trg_pad_idx, epochs)
    
    acc = evaluate_accuracy(test_dataset, model, batch_size, trg_pad_idx)
    print(f"Accuracy en el conjunto de test: {acc * 100:.2f}%")
    
    test_sentence = "Two young, White males are outside near many bushes."
    translation = translate_sentence(test_sentence, source_field, target_field, model, device)
    print("Traducción:", translation)