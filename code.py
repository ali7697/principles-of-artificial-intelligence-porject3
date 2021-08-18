from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import pad_sequence
import codecs
import nltk
nltk.download('punkt')
from nltk.lm import MLE
cut_off = 0

infile = 'E:\educational\9th Semester\AI\proj\proj3\\train_set\hafez_train.txt'
with codecs.open(infile, encoding='utf8') as f:
    raw = f.read()
# raw = raw.replace(u'\u200c', ' ')
raw = raw.replace(u'\n', ' . ')
tokenized_text = [list(pad_sequence(word_tokenize(sent), pad_left=True, left_pad_symbol="<s>", n=2))
                  for sent in sent_tokenize(raw)]
all_tokens = []
for token in tokenized_text:
    for word in token:
        all_tokens.append(word)
freq_dist = nltk.FreqDist(all_tokens)
new_tokens = [word for word in freq_dist.keys() if freq_dist[word] > cut_off]
new_tokenized_text = []
for token in tokenized_text:
    tmp = []
    for word in token:
        if word in new_tokens:
            tmp.append(word)
        # else:
        #     tmp.append('<unk>')
    new_tokenized_text.append(tmp)
n = 2
train_data, padded_sents = padded_everygram_pipeline(n, new_tokenized_text)

model_hafez = MLE(n)
model_hafez.fit(train_data, padded_sents)

infile = 'E:\educational\9th Semester\AI\proj\proj3\\train_set\\ferdowsi_train.txt'
with codecs.open(infile, encoding='utf8') as f:
    raw = f.read()
# raw = raw.replace(u'\u200c', ' ')
raw = raw.replace(u'\n', ' . ')
tokenized_text = [list(pad_sequence(word_tokenize(sent), pad_left=True, left_pad_symbol="<s>", n=2))
                  for sent in sent_tokenize(raw)]
all_tokens = []
for token in tokenized_text:
    for word in token:
        all_tokens.append(word)
freq_dist = nltk.FreqDist(all_tokens)
new_tokens = [word for word in freq_dist.keys() if freq_dist[word] > cut_off]
new_tokenized_text = []
for token in tokenized_text:
    tmp = []
    for word in token:
        if word in new_tokens:
            tmp.append(word)
        # else:
        #     tmp.append('<unk>')
    new_tokenized_text.append(tmp)
n = 2
train_data, padded_sents = padded_everygram_pipeline(n, new_tokenized_text)
model_ferdowsi = MLE(n)
model_ferdowsi.fit(train_data, padded_sents)

infile = 'E:\educational\9th Semester\AI\proj\proj3\\train_set\\molavi_train.txt'
with codecs.open(infile, encoding='utf8') as f:
    raw = f.read()
# raw = raw.replace(u'\u200c', ' ')
raw = raw.replace(u'\n', ' . ')
tokenized_text = [list(pad_sequence(word_tokenize(sent), pad_left=True, left_pad_symbol="<s>", n=2))
                  for sent in sent_tokenize(raw)]
all_tokens = []
for token in tokenized_text:
    for word in token:
        all_tokens.append(word)
freq_dist = nltk.FreqDist(all_tokens)
new_tokens = [word for word in freq_dist.keys() if freq_dist[word] > cut_off]
new_tokenized_text = []
for token in tokenized_text:
    tmp = []
    for word in token:
        if word in new_tokens:
            tmp.append(word)
        # else:
        #     tmp.append('<unk>')
    new_tokenized_text.append(tmp)
n = 2
train_data, padded_sents = padded_everygram_pipeline(n, new_tokenized_text)
model_molavi = MLE(n)
model_molavi.fit(train_data, padded_sents)


lambdas = [0.6, 0.399, 0.001]
epsilon = 0.01

def choose_group(token):
    probs_1 = []
    probs_2 = []
    probs_3 = []
    for i in range(1, len(token)):
        p_bigram = model_ferdowsi.score(token[i], token[i - 1].split())
        p_uni = model_ferdowsi.score(token[i])
        probs_1.append(lambdas[0] * p_bigram + lambdas[1] * p_uni + lambdas[2] * epsilon)

        p_bigram = model_hafez.score(token[i], token[i - 1].split())
        p_uni = model_hafez.score(token[i])
        probs_2.append(lambdas[0] * p_bigram + lambdas[1] * p_uni + lambdas[2] * epsilon)

        p_bigram = model_molavi.score(token[i], token[i - 1].split())
        p_uni = model_molavi.score(token[i])
        probs_3.append(lambdas[0] * p_bigram + lambdas[1] * p_uni + lambdas[2] * epsilon)
    ps = [1, 1, 1]
    for num in probs_1:
        ps[0] *= num
    for num in probs_2:
        ps[1] *= num
    for num in probs_3:
        ps[2] *= num
    return str(ps.index(max(ps)) + 1)


test_file = 'E:\educational\9th Semester\AI\proj\proj3\\test_set\\test_file.txt'
with codecs.open(test_file, encoding='utf8') as f:
    lines = f.readlines()
groups = []
labels = []
sum_correct = 0
total_num = len(lines)
for line in lines:
    line = line.split('\t')
    labels.append(line[0])
    # line[1] = line[1].replace(u'\u200c', ' ')
    line[1] = line[1].replace(u'\n', ' . ')
    line[1] = line[1].replace(u'\r', '')
    line[1] = list(pad_sequence(word_tokenize(line[1]), pad_left=True, left_pad_symbol="<s>", n=2))
    g = choose_group(line[1])
    groups.append(g)
    if g == line[0]:
        sum_correct += 1
print(sum_correct)
print(total_num)
print(sum_correct/total_num)