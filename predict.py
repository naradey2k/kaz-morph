import pickle

def load_model(model_path="kaz_crf_model.model"):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def prepare_word_features(word, delta=6):
    word_plus = '[' + word + ']'
    word_features = []
    
    for i in range(len(word_plus)):
        char_dic = {}
        for j in range(delta):
            char_dic['right_' + word_plus[i:i + j + 1]] = 1
        for j in range(delta):
            if i - j - 1 < 0: 
                break
            char_dic['left_' + word_plus[i - j - 1:i]] = 1
        char_dic['pos_start_' + str(i)] = 1
        if word_plus[i] in ['a', 's', 'o']:
            char_dic[str(word_plus[i])] = 1
        word_features.append(char_dic)
    
    return [word_features]

def predict_segmentation(word, model_path="kaz_crf_model.model"):
    model = load_model(model_path)
    
    X = prepare_word_features(word)
    prediction = model.predict(X)[0]
        
    word_plus = '[' + word + ']'
    
    segments = []
    current_segment = ""
    
    for i, (char, label) in enumerate(zip(word_plus, prediction)):
        if char in ['[', ']']:  
            continue
            
        current_segment += char
        
        if label in ['E', 'S']:
            segments.append(current_segment)
            current_segment = "" 
    

    if current_segment:
        segments.append(current_segment)
    
    return "/".join(segments)

if __name__ == "__main__":
    word = "қазақтар"
    print(word)
    print(f"Segmentation: {predict_segmentation(word)}")