# From https://github.com/noranta4/Supervised-Morphological-Segmentation

import sklearn_crfsuite
import pickle

training_dic = {}
dev_dic = {}

delta = 3
epsilon = 1e-4
max_iterations = 160
input_files = ['train_data.txt', 'val_data.txt'] 
dictionaries = (training_dic, dev_dic)
counter = 0
limit = 0 
n_samples = 10000

for filename in input_files:
    with open(filename, encoding='utf-8') as inputfile:
        for line in inputfile:
            word, segmentation = line.split('\t')
            result = []
            
            for morpheme in segmentation.split('/'):
                result.append(morpheme.split(':')[0])
            
            label = ''
            for morph in result:
                if len(morph) == 1:
                    label += 'S'
                else:
                    label += 'B'
                    for i in range(len(morph)-2):
                        label += 'M'
                    label += 'E'
            dictionaries[counter][word] = label
            limit += 1 # LIMIT ON
            if limit > n_samples: break
        print(f"Number of samples: {limit} in {filename}")
        limit = 0
        counter += 1

def prepare_data(word_dictonary, delta):
    X = [] # list (learning set) of list (word) of dics (chars), INPUT for crf
    Y = [] # list (learning set) of list (word) of labels (chars), INPUT for crf
    words = [] # list (learning set) of list (word) of chars
    for word in word_dictonary:
        word_plus = '[' + word + ']' # <w> and <\w> replaced with [ and ]
        word_list = [] # container of the dic of each character in a word
        word_label_list = [] # container of the label of each character in a word
        for i in range(len(word_plus)):
            char_dic = {} # dic of features of the actual char
            for j in range(delta):
                char_dic['right_' + word_plus[i:i + j + 1]] = 1
            for j in range(delta):
                if i - j - 1 < 0: break
                char_dic['left_' + word_plus[i - j - 1:i]] = 1
            char_dic['pos_start_' + str(i)] = 1  # extra feature: left index of the letter in the word
            # char_dic['pos_end_' + str(len(word) - i)] = 1  # extra feature: right index of the letter in the word
            if word_plus[i] in ['a', 's', 'o']: # extra feature: stressed characters (discussed in the report)
                char_dic[str(word_plus[i])] = 1
            word_list.append(char_dic)

            if word_plus[i] == '[': word_label_list.append('[') # labeling start and end
            elif word_plus[i] == ']': word_label_list.append(']')
            else: word_label_list.append(word_dictonary[word][i-1]) # labeling chars
        X.append(word_list)
        Y.append(word_label_list)
        temp_list_word = [char for char in word_plus]
        words.append(temp_list_word)
    return (X, Y, words)

print('Features computed')

X_training, Y_training, words_training = prepare_data(training_dic, delta)
X_dev, Y_dev, words_dev = prepare_data(dev_dic, delta)

crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)
pickle.dump(crf, open("kaz_crf_model.model", "wb"))

print('Training done')

Y_predict = crf.predict(X_dev)
H, I, D = 0, 0, 0
for j in range(len(Y_dev)):
    for i in range(len(Y_dev[j])):
        if Y_dev[j][i] == 'E' or Y_dev[j][i] == 'S':
            if Y_dev[j][i] == Y_predict[j][i]:
                H += 1
            else:
                D += 1
        else:
            if (Y_predict[j][i] == 'E' or Y_predict[j][i] == 'S'):
                I += 1

if (H + I) == 0:
    P = 0.0
    print("Warning: No boundary positions predicted by model (H+I=0)")
else:
    P = float(H)/(H+I)

if (H + D) == 0:
    R = 0.0
    print("Warning: No boundary positions in ground truth (H+D=0)")
else:
    R = float(H)/(H+D)

if (P + R) == 0:
    F1 = 0.0
    print("Warning: Both Precision and Recall are 0")
else:
    F1 = (2*P*R)/(P+R)

print('Delta = ' + str(delta) + '\tNsamples = ' + str(n_samples) + '\tepsilon = ' + str(epsilon) + '\tmax_iter = ' + str(max_iterations))
print('Precision = ' + str(P))
print('Recall = ' + str(R))
print('F1-score = ' + str(F1))


