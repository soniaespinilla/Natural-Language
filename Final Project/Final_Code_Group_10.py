import nltk
import re
import json
from nltk.tokenize import WordPunctTokenizer
from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report, confusion_matrix

# Download punkt
nltk.download('punkt', quiet=True)

# ---------------- Constants ----------------
class Constants:
    NEGATION_LABEL = "NEG"
    UNCERTAINTY_LABEL = "UNC"
    NEGATION_SCOPE_LABEL = "NSCO"
    UNCERTAINTY_SCOPE_LABEL = "USCO"
    TRAIN_JSON = 'negacio_train_v2024.json'
    TEST_JSON = 'negacio_test_v2024.json'

# ---------------- Data Loading ----------------
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------- Rule-based System ----------------
def get_word_list_from_data(json_file):
    documents = load_data(json_file)
    negation_word_set = set()
    uncertainty_word_set = set()
    for doc in documents:
        text = doc['data']['text']
        for pred in doc.get('predictions', []):
            for ann in pred.get('result', []):
                labels = ann['value']['labels']
                start, end = ann['value']['start'], ann['value']['end']
                word = text[start:end]
                if Constants.NEGATION_LABEL in labels:
                    negation_word_set.add(word)
                elif Constants.UNCERTAINTY_LABEL in labels:
                    uncertainty_word_set.add(word)
    return list(negation_word_set), list(uncertainty_word_set)

negation_word_list, uncertainty_word_list = get_word_list_from_data(Constants.TRAIN_JSON)

# ---------------- Core Regex & Scope Extraction Function ----------------
def find_word_positions(text, negation_word_list, uncertainty_word_list,
                        label_neg, label_uncertainty,
                        label_scope_neg, label_scope_uncertainty):
    """
    Processes the text to find negation and uncertainty tokens, and then extracts their scopes.
    The function uses regex and tokenization so that scope boundaries are determined by stop words,
    punctuation, and custom comma logic.
    """
    negation_set = set(re.escape(word.lower()) for word in negation_word_list)
    uncertainty_set = set(re.escape(word.lower()) for word in uncertainty_word_list)
    stop_phrases = set([
        "que", "si", "como", "com", "siempre y cuando", "sempre i quan",
        "porque", "perque", "pues", "doncs", "ya que", "ja que", "dado que",
        "donat que", "puesto que", "visto que", "a causa de que", "luego",
        "asi que", "així que", "de modo que", "de manera que", "de forma que",
        "conque", "aunque", "si bien", "si be", "aun cuando", "encara quan",
        "por mas que", "per mes que", "por mucho que", "asi", "luego que",
        "mientras", "mentre", "mentres", "para que", "per a que", "a fin de que",
        "pero", "por lo que", "pel que"
    ])
    stop_punctuation = set([r'\.', r'\)', r'\(', r':', r',', r';', r'!', r'\?', r'-'])

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    spans = list(tokenizer.span_tokenize(text))
    result = []
    entity_count = 0
    covered_tokens = set()

    def check_comma_condition(current_index):
        stop_due_to_negation = False
        j = current_index
        while j + 1 < len(tokens):
            next_token = tokens[j + 1]
            next_token_normalized = re.sub(r'[^\w\s]', '', next_token).lower()
            if next_token_normalized in {'i', 'y', 'o', 'ni', 'tampoco', 'e'}:
                return (False, False)
            if next_token_normalized in negation_set or next_token_normalized in uncertainty_set:
                return (True, True)
            if any(re.search(pattern, next_token) for pattern in stop_punctuation):
                return (True, False)
            j += 1
        return (True, False)

    def create_entity(start, end, label):
        nonlocal entity_count
        entity = {
            "value": {"start": start, "end": end, "labels": [label]},
            "id": f"ent{entity_count}",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
        }
        entity_count += 1
        return entity

    def process_word_match(i, start, end, token_normalized, word_set, label):
        if i in covered_tokens:
            return False
        if not re.match(r'(' + '|'.join(word_set) + r')\b', token_normalized):
            return False

        # Create entity for cue
        main_entity = create_entity(start, end, label)
        result.append(main_entity)
        covered_tokens.add(i)

        # Initialize scope
        scope_start = end
        scope_end = end
        j = i
        steps = 0
        preposition_tokens = {
            "se": r'\b\w+(?:ar|er|re|ir|or|ur)se\b',
            "le": r'\b\w+(?:ar|er|re|ir|or|ur)le\b',
            "li": r'\b\w+(?:ar|er|re|ir|or|ur)li\b'
        }
        immediate_preposition = None

        def is_stop_token(idx):
            tk = tokens[idx]
            norm = re.sub(r'[^\w\s]', '', tk).lower()
            if any(re.fullmatch(regex, norm) for regex in preposition_tokens.values()):
                return False
            if norm in stop_phrases or any(re.search(p, tk) for p in stop_punctuation):
                return True
            if re.match(r'\d+', tk):
                return True
            if tk == ',':
                sc, _ = check_comma_condition(idx)
                return sc
            return False

        # Backward scope detection
        if i + 1 < len(tokens) and is_stop_token(i + 1):
            scope_end = start
            scope_start = start
            k = i - 1
            while k >= 0:
                if is_stop_token(k):
                    scope_start = spans[k][0]
                    k -= 1
                else:
                    scope_start = spans[k][0]
                    break
            while scope_start < scope_end and text[scope_start] == ' ':
                scope_start += 1
            if scope_end > scope_start:
                sco_label = label_scope_uncertainty if label == label_uncertainty else label_scope_neg
                result.append(create_entity(scope_start, scope_end, sco_label))
                for idx, (s, e) in enumerate(spans):
                    if s >= scope_start and e <= scope_end:
                        covered_tokens.add(idx)
            return True

        # Forward scope detection
        while j + 1 < len(spans):
            steps += 1
            nt = tokens[j + 1]
            ns, ne = spans[j + 1]
            ntn = re.sub(r'[^\w\s]', '', nt).lower()

            if steps == 1 and ntn in preposition_tokens:
                immediate_preposition = ntn

            if immediate_preposition:
                stop_prep = (ntn != immediate_preposition and re.fullmatch(preposition_tokens[immediate_preposition], ntn))
            else:
                stop_prep = any(steps > 1 and re.fullmatch(pat, ntn) for pat in preposition_tokens.values())

            punct = any(re.search(pat, nt) for pat in stop_punctuation)
            num = bool(re.match(r'\d+', nt))
            comma = (nt == ',')

            if comma:
                stop_cond, stop_neg = check_comma_condition(j + 1)
            else:
                stop_neg = (ntn in negation_set or ntn in uncertainty_set)
                stop_cond = stop_neg or (ntn in stop_phrases or punct or stop_prep or num)

            if stop_cond:
                if stop_neg:
                    scope_end = ns
                elif num and j > 0:
                    prev_ns, _ = spans[j]
                    scope_end = prev_ns - (1 if text[prev_ns - 1] == ' ' else 0)
                else:
                    scope_end = ns - (1 if text[ns - 1] == ' ' else 0)
                break
            else:
                scope_end = ne
                j += 1

        if scope_end > scope_start:
            sco_label = label_scope_uncertainty if label == label_uncertainty else label_scope_neg
            result.append(create_entity(scope_start, scope_end, sco_label))
            for idx, (s, e) in enumerate(spans):
                if s >= scope_start and e <= scope_end:
                    covered_tokens.add(idx)
        return True

    # Process all tokens
    for i, ((s, e), tok) in enumerate(zip(spans, tokens)):
        if i in covered_tokens:
            continue
        norm = re.sub(r'[^\w\s]', '', tok).lower()
        process_word_match(i, s, e, norm, negation_set, Constants.NEGATION_LABEL)
        process_word_match(i, s, e, norm, uncertainty_set, Constants.UNCERTAINTY_LABEL)

    return {"result": result}

# ---------------- Visualization ----------------
def visualize_scopes(text, entities):
    highlighted = list(text)
    for ent in entities:
        st, ed = ent['value']['start'], ent['value']['end']
        lab = ent['value']['labels'][0]
        if lab == Constants.NEGATION_LABEL:
            col = 'yellow'
        elif lab == Constants.UNCERTAINTY_LABEL:
            col = 'blue'
        elif lab == Constants.NEGATION_SCOPE_LABEL:
            col = 'green'
        elif lab == Constants.UNCERTAINTY_SCOPE_LABEL:
            col = 'magenta'
        else:
            col = None
        for idx in range(st, ed):
            char = highlighted[idx]
            if col:
                highlighted[idx] = colored('[', col) + char + colored(']', col)
            else:
                highlighted[idx] = f'[{char}]'
    print(''.join(highlighted))

# ---------------- Accuracy & Prediction ----------------
def calculate_accuracy(preds, annots, text):
    tok = WordPunctTokenizer()
    scores = []
    for a in annots['result']:
        ast, aed = a['value']['start'], a['value']['end']
        lab = a['value']['labels'][0]
        gt = text[ast:aed].lower().strip()
        gts = tok.tokenize(gt)
        if not gts:
            continue
        best = 0
        for p in preds[0]['result']:
            if lab in p['value']['labels']:
                pst, ped = p['value']['start'], p['value']['end']
                pt = text[pst:ped].lower().strip()
                pts = tok.tokenize(pt)
                if not pts:
                    continue
                overlap = sum(1 for t in gts if t in pts)
                best = max(best, overlap / len(gts))
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0

def run_rule(docs):
    outputs = []
    for doc in docs:
        out = find_word_positions(
            doc['data']['text'],
            negation_word_list, uncertainty_word_list,
            Constants.NEGATION_LABEL, Constants.UNCERTAINTY_LABEL,
            Constants.NEGATION_SCOPE_LABEL, Constants.UNCERTAINTY_SCOPE_LABEL
        )
        outputs.append(out)
    return outputs

# ---------------- CRF-based ML System (Spanish, enhanced ML features) ----------------
import json
import numpy as np
import fasttext, fasttext.util
import spacy
from nltk.tokenize import WordPunctTokenizer
from sklearn.cluster import KMeans
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Data Loading ----------------

negation_word_list, uncertainty_word_list = get_word_list_from_data(Constants.TRAIN_JSON)
neg_lex = set(negation_word_list)
unc_lex = set(uncertainty_word_list)

# ---------------- FastText & spaCy Setup ----------------
fasttext.util.download_model('es', if_exists='ignore')
ft_model = fasttext.load_model('cc.es.300.bin')
nlp = spacy.load('es_core_news_sm')

# ---------------- Clustering ----------------
def build_clusters(train_docs, n_clusters=50):
    toks = set()
    wp = WordPunctTokenizer()
    for doc in train_docs:
        toks.update(t.lower() for t in wp.tokenize(doc['data']['text']))
    toks = list(toks)
    vecs = np.vstack([ft_model.get_word_vector(t)[:50] for t in toks])
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(vecs)
    return {toks[i]: int(km.labels_[i]) for i in range(len(toks))}

cluster_map = {}

# ---------------- Feature Extraction ----------------
def word_shape(text):
    return ''.join(
        'X' if c.isupper() else
        'x' if c.islower() else
        'd' if c.isdigit() else
        'p'
        for c in text
    )

def word2features(tokens, idx):
    tok = tokens[idx]
    txt = tok.text.lower()
    feats = {
        'bias': 1.0,
        'word.lower': txt,
        'lemma': tok.lemma_,
        'pos': tok.pos_,
        'dep': tok.dep_,
        'head.lower': tok.head.text.lower(),
        'in_neg_lex': txt in neg_lex,
        'in_unc_lex': txt in unc_lex,
        'shape': word_shape(tok.text),
        'prefix3': txt[:3],
        'suffix3': txt[-3:],
        'cluster': cluster_map.get(txt, -1)
    }
    for off in (-2, -1, 1, 2):
        j = idx + off
        if 0 <= j < len(tokens):
            w = tokens[j]
            feats[f'{off}:word.lower'] = w.text.lower()
            feats[f'{off}:pos']       = w.pos_
        else:
            feats[f'{off}:OOB'] = True

    vec50 = ft_model.get_word_vector(txt)[:50]
    for i, v in enumerate(vec50):
        feats[f'emb{i}'] = float(v)

    return feats

# ---------------- Data Preparation ----------------
def prepare_cue_data(docs):
    X, Y = [], []
    for doc in docs:
        sp = nlp(doc['data']['text'])
        toks = list(sp)
        spans = [(t.idx, t.idx+len(t.text)) for t in toks]
        labels = ['O'] * len(toks)
        for pred in doc.get('predictions', []):
            for ent in pred['result']:
                st, ed = ent['value']['start'], ent['value']['end']
                is_cue = ent['value']['labels'][0] in {Constants.NEGATION_LABEL, Constants.UNCERTAINTY_LABEL}
                for i, (s, e) in enumerate(spans):
                    if s >= st and e <= ed and is_cue:
                        labels[i] = ('B-' if s == st else 'I-') + 'CUE'
        X.append([word2features(toks, i) for i in range(len(toks))])
        Y.append(labels)
    return X, Y

def prepare_scope_data(docs):
    X, Y = [], []
    for doc in docs:
        sp = nlp(doc['data']['text'])
        toks = list(sp)
        spans = [(t.idx, t.idx+len(t.text)) for t in toks]
        scope_spans = [
            (ent['value']['start'], ent['value']['end'])
            for pred in doc.get('predictions', [])
            for ent in pred['result']
            if ent['value']['labels'][0] in {Constants.NEGATION_SCOPE_LABEL, Constants.UNCERTAINTY_SCOPE_LABEL}
        ]
        labels, in_scope = [], False
        for t in toks:
            inside = any(s <= t.idx < e for s, e in scope_spans)
            if inside:
                labels.append('B-SCOPE' if not in_scope else 'I-SCOPE')
                in_scope = True
            else:
                labels.append('O')
                in_scope = False
        X.append([word2features(toks, i) for i in range(len(toks))])
        Y.append(labels)
    return X, Y

# ---------------- Hyperparameter Grid Search ----------------
def grid_search_crf(X, Y):
    param_grid = {
        'algorithm': ['lbfgs'],
        'c1': [0.1, 0.3],
        'c2': [0.01, 0.1],
        'max_iterations': [200, 300]
    }
    best_params, best_score = None, -1
    for params in tqdm(list(ParameterGrid(param_grid)), desc="Grid search CRF"):
        crf = CRF(**params, all_possible_transitions=True)
        crf.fit(X, Y)
        y_pred = crf.predict(X)
        flat_t = [l for seq in Y for l in seq]
        flat_p = [l for seq in y_pred for l in seq]
        score = recall_score(
            flat_t, flat_p, average='macro',
            labels=[l for l in set(flat_t) if l.startswith('I-')]
        )
        if score > best_score:
            best_score, best_params = score, params
    return best_params

# ---------------- Two-Stage CRF ----------------
def run_two_stage(train_docs, test_docs):
    global cluster_map
    cluster_map = build_clusters(train_docs, n_clusters=50)

    Xc_tr, Yc_tr = prepare_cue_data(train_docs)
    Xc_te, Yc_te = prepare_cue_data(test_docs)
    Xs_tr, Ys_tr = prepare_scope_data(train_docs)
    Xs_te, Ys_te = prepare_scope_data(test_docs)

    cue_params = grid_search_crf(Xc_tr, Yc_tr)
    cue_crf = CRF(**cue_params, all_possible_transitions=True)
    cue_crf.fit(Xc_tr, Yc_tr)

    cue_pred = []
    for x in tqdm(Xc_te, desc="Predicting cues"):
        cue_pred.extend(cue_crf.predict([x]))

    scope_params = grid_search_crf(Xs_tr, Ys_tr)
    scope_crf = CRF(**scope_params, all_possible_transitions=True)
    scope_crf.fit(Xs_tr, Ys_tr)

    # predict scopes with progress bar
    scope_pred = []
    for x in tqdm(Xs_te, desc="Predicting scopes"):
        scope_pred.extend(scope_crf.predict([x]))

    # Evaluate cue stage
    flat_true = [l for seq in Yc_te for l in seq]
    flat_pred = [l for seq in cue_pred for l in seq]
    print("=== Cue CRF Classification Report ===")
    print(classification_report(flat_true, flat_pred))
    cm = confusion_matrix(flat_true, flat_pred, labels=['B-CUE','I-CUE','O'])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['B','I','O'], yticklabels=['B','I','O'])
    plt.title("Cue Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Evaluate scope stage
    flat_true = [l for seq in Ys_te for l in seq]
    flat_pred = [l for seq in scope_pred for l in seq]
    print("=== Scope CRF Classification Report ===")
    print(classification_report(flat_true, flat_pred))
    cm = confusion_matrix(flat_true, flat_pred, labels=['B-SCOPE','I-SCOPE','O'])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['B','I','O'], yticklabels=['B','I','O'])
    plt.title("Scope Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return cue_crf, scope_crf

# ---------------- Deep Learning System ----------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn.functional as F
from sklearn_crfsuite import metrics as crf_metrics
import sklearn_crfsuite.metrics as crf_metrics


tags_dl = ['O', 'B-NEG','I-NEG','B-UNC','I-UNC','B-NSCO','I-NSCO','B-USCO','I-USCO']


def get_casing_vector(token):
    """
    Returns an 8-dim one-hot vector for casing feature of a token.
    Categories:
    0: all lowercase
    1: all uppercase
    2: initial uppercase (title case)
    3: contains digit(s)
    4: contains hyphen(s)
    5: other (mixed case)
    6: all punctuation
    7: numeric (all digits)
    """
    import string

    vec = [0]*8
    if token.isdigit():
        vec[7] = 1
    elif all(c in string.punctuation for c in token):
        vec[6] = 1
    elif token.islower():
        vec[0] = 1
    elif token.isupper():
        vec[1] = 1
    elif token.istitle():
        vec[2] = 1
    elif any(c.isdigit() for c in token):
        vec[3] = 1
    elif '-' in token:
        vec[4] = 1
    else:
        vec[5] = 1
    return torch.tensor(vec, dtype=torch.float)


from collections import defaultdict
from nltk.tokenize import WordPunctTokenizer

class DLSequenceDataset(Dataset):
    def __init__(self, docs, char_vocab=None, max_word_len=20):
        self.examples = []
        self.max_word_len = max_word_len
        self.tokenizer = WordPunctTokenizer()

        # Build or use provided char vocabulary
        if char_vocab is None:
            self.char2idx = {'<pad>': 0, '<unk>': 1}
            for doc in docs:
                text = doc['data']['text']
                toks = self.tokenizer.tokenize(text)
                for tok in toks:
                    for c in tok:
                        if c not in self.char2idx:
                            self.char2idx[c] = len(self.char2idx)
        else:
            self.char2idx = char_vocab
        print(self.char2idx)

        # Build PoS vocabulary (universal tags from spaCy)
        self.pos2idx = {'<pad>': 0}
        for doc in docs:
            text = doc['data']['text']
            spacy_doc = nlp(text)
            for token in spacy_doc:
                if token.pos_ not in self.pos2idx:
                    self.pos2idx[token.pos_] = len(self.pos2idx)

        counts = {lbl: 0 for lbl in tags_dl}

        for doc_idx, doc in enumerate(docs):
            text = doc['data']['text']
            toks = self.tokenizer.tokenize(text)
            offs = list(self.tokenizer.span_tokenize(text))
            labels = ['O'] * len(toks)

            # Assign IOB labels for entities
            for pred in doc.get('predictions', []):
                for ent in pred.get('result', []):
                    st, ed, lb = ent['value']['start'], ent['value']['end'], ent['value']['labels'][0]
                    for i, (s, e) in enumerate(offs):
                        if s >= st and e <= ed:
                            labels[i] = 'B-'+lb if s == st else 'I-'+lb
            for l in labels:
                counts[l] += 1

            # Word embeddings (e.g., fastText)
            vecs = [torch.tensor(ft_model.get_word_vector(t.lower()), dtype=torch.float) for t in toks]

            # Label indices
            idxs = [tags_dl.index(l) for l in labels]

            # Char-level tensor: [seq_len, max_word_len]
            char_tensors = []
            for tok in toks:
                char_ids = [self.char2idx.get(c, self.char2idx['<unk>']) for c in tok[:self.max_word_len]]
                padded = char_ids + [0] * (self.max_word_len - len(char_ids))
                char_tensors.append(torch.tensor(padded, dtype=torch.long))

            # Get PoS tags from spaCy and map to pos2idx
            spacy_doc = nlp(text)
            pos_tags = [self.pos2idx.get(token.pos_, 0) for token in spacy_doc]

            # Align pos_tags length to tokens length
            if len(pos_tags) > len(toks):
                pos_tags = pos_tags[:len(toks)]
            elif len(pos_tags) < len(toks):
                pos_tags += [0] * (len(toks) - len(pos_tags))

            pos_tensor = torch.tensor(pos_tags, dtype=torch.long)

            # Casing one-hot vectors
            casing_vectors = [get_casing_vector(t) for t in toks]
            casing_tensor = torch.stack(casing_vectors)  # [seq_len, 8]

            self.examples.append((
                torch.stack(vecs),        # [seq_len, 300]
                torch.tensor(idxs, dtype=torch.long),
                torch.stack(char_tensors),  # [seq_len, max_word_len]
                pos_tensor,                 # [seq_len]
                casing_tensor               # [seq_len, 8]
            ))

        # Class weights for loss balancing
        freqs = torch.tensor([counts[l] for l in tags_dl], dtype=torch.float)
        weights = 1.0 / (freqs + 1)
        self.class_weights = weights / weights.sum() * len(tags_dl)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(batch):
        seqs, labs, chars, poses, casings = zip(*batch)
        return (
            pad_sequence(seqs, batch_first=True),
            pad_sequence(labs, batch_first=True, padding_value=0),
            pad_sequence(chars, batch_first=True),
            pad_sequence(poses, batch_first=True, padding_value=0),
            pad_sequence(casings, batch_first=True, padding_value=0.0)
        )



class DLBiLSTM(nn.Module):
    def __init__(self, word_emb_dim, char_vocab_size, pos_vocab_size,
                 char_emb_dim=50, char_out_dim=30, pos_emb_dim=50,
                 hidden_dim=150, tagset_size=9,
                 lstm_layers=1, lstm_dropout=0.5, lstm_recur_dropout=0.25):
        super().__init__()

        # CharCNN encoder
        self.char_cnn = CharCNNEncoder(char_vocab_size, char_emb_dim, char_out_dim)

        # PoS tag embedding
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_emb_dim, padding_idx=0)

        # Casing categories: 8-dim one-hot vector per token
        self.casing_dim = 8

        # Final input dim to LSTM = word_emb + char_out + pos_emb + casing_dim
        lstm_input_dim = word_emb_dim + char_out_dim + pos_emb_dim + self.casing_dim

        # LSTM with multiple layers, dropout only if layers > 1
        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )
        # For simplicity, we just use dropout between layers.
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, word_embs, char_ids, pos_ids, casing_ids):
        # word_embs: [batch, seq_len, word_emb_dim]
        # char_ids: [batch, seq_len, max_word_len]
        # pos_ids: [batch, seq_len]
        # casing_ids: [batch, seq_len, casing_dim]
        char_feats = self.char_cnn(char_ids)  # [batch, seq_len, char_out_dim]

        # pos_ids: [batch, seq_len]
        pos_embs = self.pos_embedding(pos_ids)  # [batch, seq_len, pos_emb_dim]

        # Concatenar todas las features por token
        x = torch.cat([word_embs, char_feats, pos_embs, casing_ids], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # Proyección final a logits
        out = self.fc(lstm_out)

        return out

class CharCNNEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim=50, num_filters=30, kernel_size=3):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_emb_dim, num_filters, kernel_size=kernel_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, char_seq_batch):
        # char_seq_batch: [batch, seq_len, word_len]
        batch_size, seq_len, word_len = char_seq_batch.size()
        char_seq = char_seq_batch.view(-1, word_len)  # flatten for embedding
        emb = self.char_embedding(char_seq)           # [batch*seq_len, word_len, char_emb_dim]
        emb = emb.transpose(1, 2)                      # [batch*seq_len, char_emb_dim, word_len]
        conv_out = self.conv(emb)                      # [batch*seq_len, num_filters, L]
        conv_out = F.relu(conv_out)
        pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)  # [batch*seq_len, num_filters]
        pooled = self.dropout(pooled)
        return pooled.view(batch_size, seq_len, -1)  # reshape back [batch, seq_len, num_filters]

def run_dl(train_docs, test_docs):
    train_ds = DLSequenceDataset(train_docs)
    test_ds = DLSequenceDataset(test_docs, char_vocab=train_ds.char2idx)

    model = DLBiLSTM(
        word_emb_dim=ft_model.get_dimension(),
        char_vocab_size=len(train_ds.char2idx),
        pos_vocab_size=len(train_ds.pos2idx),
        char_emb_dim=50,
        char_out_dim=30,
        pos_emb_dim=50,
        hidden_dim=150,
        tagset_size=len(tags_dl)
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=DLSequenceDataset.collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()  

    for epoch in range(20):
        total_loss = 0.0
        model.train()
        for i, (seqs, labs, chars, poses, casings) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            out = model(seqs, chars, poses, casings)
            loss = loss_fn(out.view(-1, out.size(-1)), labs.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f" DL Epoch {epoch+1} loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=DLSequenceDataset.collate_fn)
    for seqs, labs, chars, poses, casings in tqdm(test_loader, desc="DL Evaluation"):
        with torch.no_grad():
            scores = model(seqs, chars, poses, casings)[0]  # batch=1
            preds = scores.argmax(dim=-1).tolist()
        true = labs.tolist()[0]

        y_true.append([tags_dl[i] for i in true])
        y_pred.append([tags_dl[i] for i in preds])

    print("Entity-level classification report:")
    print(crf_metrics.flat_classification_report(y_true, y_pred, labels=tags_dl, digits=4))

    f1 = crf_metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=tags_dl)
    print(f"Weighted F1-score: {f1:.4f}")

    # Flatten lists for scikit-learn classification report and confusion matrix
    labels_flat = [l for seq in y_true for l in seq]
    preds_flat = [p for seq in y_pred for p in seq]

    report = classification_report(labels_flat, preds_flat, labels=tags_dl, digits=4)
    cm = confusion_matrix(labels_flat, preds_flat, labels=tags_dl)

    return report, cm, tags_dl


# ---------------- Main ----------------
def main():
    train_docs = load_data(Constants.TRAIN_JSON)
    test_docs = load_data(Constants.TEST_JSON)

    print("Running CRF system...")
    cue_crf, scope_crf = run_two_stage(train_docs, test_docs)
    print(cue_crf)
    print(scope_crf)

    print("\nRunning Deep Learning system...")
    dl_report, dl_cm, dl_labels = run_dl(train_docs, test_docs)
    print(dl_report)
    print(dl_labels)
    for row in dl_cm: print(row)

if __name__ == '__main__':
    main()
