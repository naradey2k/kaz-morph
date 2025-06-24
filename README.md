# Kazakh Morphological Segmentation Using Conditional Random Fields

This project uses the implementation of sklearn-CRF from [noranta4/Supervised-Morphological-Segmentation](https://github.com/noranta4/Supervised-Morphological-Segmentation) to segment morphemes in Kazakh words.

### Data Collection and Processing

At first, some text was collected and then segmented with the supervisor and written in `.txt` format:


Each line consists of a word and its segmented morphemes in the format `morpheme:TAG`, separated by slashes (`/`). Words and their segmentations are separated by a tab (`\t`).

### Dataset Statistics

| Dataset         | # of Words |
|-----------------|------------|
| `train_data.txt`| 4689       |
| `val_data.txt`  | 521        |

The dataset was split using an 80/20 train-validation split ratio.

### Model

The model is trained using a **Conditional Random Field (CRF)**, with features extracted from each character in the word, such as:

- Character identity
- Position in the word
- Prefix/suffix patterns
- Whether the character is a vowel or consonant
- Context characters (previous/next)

### Usage

To train and evaluate the model:

```bash
python train_crf.py --train_file data/train_data.txt --val_file data/val_data.txt
