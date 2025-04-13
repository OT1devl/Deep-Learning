import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import spacy

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

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
            tokens = self.tokenize(sentence)
            for tok in tokens:
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

def load_csv_data(filepath):
    src_sentences = []
    trg_sentences = []
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
        src_numericalized = self.src_field.numericalize(self.src_sentences[idx])
        trg_numericalized = self.trg_field.numericalize(self.trg_sentences[idx])
        return {"src": src_numericalized, "trg": trg_numericalized}

class Batch:
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

def pad_sequences(sequences, pad_value):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.LongTensor(padded_seqs)

def create_batches(dataset, batch_size, pad_idx, shuffle=True):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i + batch_size]
        src_batch, trg_batch = [], []
        for idx in batch_indices:
            example = dataset[idx]
            src_batch.append(example["src"])
            trg_batch.append(example["trg"])
        src_padded = pad_sequences(src_batch, pad_idx).transpose(0, 1)
        trg_padded = pad_sequences(trg_batch, pad_idx).transpose(0, 1)
        yield Batch(src_padded, trg_padded)

source_field = Field(tokenize=tokenize_eng, init_token="<sos>", eos_token="<eos>", lower=True)
target_field = Field(tokenize=tokenize_ger, init_token="<sos>", eos_token="<eos>", lower=True)

train_src, train_trg = load_csv_data(r"")
test_src, test_trg   = load_csv_data(r"")

train_dataset = TranslationDataset(train_src, train_trg, source_field, target_field)
test_dataset  = TranslationDataset(test_src, test_trg, source_field, target_field)

source_field.build_vocab(train_src)
target_field.build_vocab(train_trg)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, batch)
        embedding = self.dropout(self.embedding(x))  # (seq_length, batch, embedding_size)
        encoder_states, (hidden, cell) = self.rnn(embedding)
        # hidden, cell: (num_layers * 2, batch, hidden_size)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_size*2)
        cell_cat   = torch.cat((cell[-2], cell[-1]), dim=1)      # (batch, hidden_size*2)
        hidden = self.fc_hidden(hidden_cat).unsqueeze(0)  # (1, batch, hidden_size)
        cell = self.fc_cell(cell_cat).unsqueeze(0)        # (1, batch, hidden_size)
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        #  Input to RNN: [context (hidden_size*2) + embedding]
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        # x: (batch,) -> (1, batch)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))  # (1, batch, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_last = hidden[-1]  # (batch, hidden_size)
        h_reshaped = h_last.unsqueeze(0).repeat(sequence_length, 1, 1)  # (seq_length, batch, hidden_size)

        energy_input = torch.cat((h_reshaped, encoder_states), dim=2)  # (seq_length, batch, hidden_size*3)
        energy = self.relu(self.energy(energy_input))  # (seq_length, batch, 1)
        attention = self.softmax(energy)  # (seq_length, batch, 1) – softmax sobre dim=0

        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)  # (1, batch, hidden_size*2)

        rnn_input = torch.cat((context_vector, embedding), dim=2)  # (1, batch, hidden_size*2 + embedding_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # outputs: (1, batch, hidden_size)
        predictions = self.fc(outputs).squeeze(0)  # (batch, output_size)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(target_field.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(source.device)
        encoder_states, hidden, cell = self.encoder(source)
        hidden = hidden.repeat(self.decoder.num_layers, 1, 1)
        cell = cell.repeat(self.decoder.num_layers, 1, 1)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

num_epochs = 20
learning_rate = 0.001
batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(source_field.vocab)
input_size_decoder = len(target_field.vocab)
output_size = len(target_field.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = target_field.vocab.stoi[target_field.pad_token]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train(dataset, epochs, batch_size, pad_idx):
    model.train()
    batch_total = len(dataset) / batch_size
    for epoch in range(epochs):
        print(f'Epoch: [{epoch+1}/{epochs}]')
        for batch_idx, batch in enumerate(create_batches(dataset, batch_size, pad_idx, shuffle=True)):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)
            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            print(f'[{batch_idx}/{batch_total:.2f}] Batch Loss: {loss.item():.4f}')

def evaluate_accuracy(dataset, batch_size, pad_idx):
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for batch in create_batches(dataset, batch_size, pad_idx, shuffle=False):
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            output = model(src, trg, teacher_force_ratio=0.0)
            predictions = output.argmax(2)
            for t in range(1, trg.shape[0]):
                true_tokens = trg[t]
                pred_tokens = predictions[t]
                mask = (true_tokens != pad_idx)
                total_tokens += mask.sum().item()
                correct_tokens += ((pred_tokens == true_tokens) & mask).sum().item()
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return accuracy

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    tokens = src_field.tokenize(sentence.lower())
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    numericalized = [src_field.vocab.stoi.get(tok, src_field.vocab.stoi[src_field.pad_token]) for tok in tokens]
    sentence_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)  # (seq_len, 1)
    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor)

    hidden = hidden.repeat(model.decoder.num_layers, 1, 1)
    cell = cell.repeat(model.decoder.num_layers, 1, 1)
    outputs = [trg_field.vocab.stoi[trg_field.init_token]]
    for _ in range(max_len):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_states, hidden, cell)
        best_guess = output.argmax(1).item()
        outputs.append(best_guess)
        if best_guess == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    translated_tokens = [trg_field.vocab.itos[idx] for idx in outputs]
    return " ".join(translated_tokens[1:-1])

def save_model(model, path=r"\RNN\Seq2Seq\checkpoints\seq2seq_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en: {path}")

if __name__ == "__main__":
    train(train_dataset, num_epochs, batch_size, pad_idx)
    
    accuracy = evaluate_accuracy(test_dataset, batch_size, pad_idx)
    print(f"Accuracy en el conjunto de test: {accuracy * 100:.2f}%")
    
    test_sentence = "Two young, White males are outside near many bushes."
    translation = translate_sentence(test_sentence, source_field, target_field, model, device)
    print("Traducción:", translation)
    
    save_model(model, r"\RNN\Seq2Seq\checkpoints\seq2seq_model.pth")
