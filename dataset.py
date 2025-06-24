import string
from sklearn.model_selection import train_test_split

def is_number(word):
    try:
        float(word)
        return True
    except ValueError:
        return False

def contains_number(word):
    """Check if a word contains any digits"""
    return any(char.isdigit() for char in word)


def text_to_data(test_size=0.2):
    processed_data = []
    with open("data.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip().split()
            for word in line:
                if word == "–" or word == "—":
                    continue
                if is_number(word):
                    continue
                if contains_number(word):
                    continue
                word_clean = word.replace("\u00B7", "")
                word_clean = word_clean.translate(str.maketrans('', '', string.punctuation)).replace("«", "").replace("»", "")
                segmentation = word.translate(str.maketrans('', '', string.punctuation)).replace("«", "").replace("»", "")
                morphs = segmentation.split("\u00B7")
                new_segmentation = []

                for i, morph in enumerate(morphs):
                    if i == 0:
                        new_segmentation.append(f"{morph}:ROOT")
                    else:
                        new_segmentation.append(f"{morph}:MORPH")
                
                line = f"{word_clean}\t{'/'.join(new_segmentation)}"
                processed_data.append(line)
    
    train_data, val_data = train_test_split(
        processed_data, 
        test_size=test_size,
        random_state=42,
        shuffle=True
    )
    
    with open("train_data.txt", 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')
    
    with open("val_data.txt", 'w', encoding='utf-8') as f:
        for line in val_data:
            f.write(line + '\n')
    
    print("\nDataset Statistics:")
    print(f"Total words: {len(processed_data)}")
    
    print(f"\nTraining set:")
    print(f"Number of words: {len(train_data)}")
    
    print(f"\nValidation set:")
    print(f"Number of words: {len(val_data)}")

if __name__ == "__main__":
    text_to_data(test_size=0.1)