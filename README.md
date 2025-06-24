# Kazakh Morphological Segmentation Using Conditional Random Fields

This project uses the implementation of sklearn-CRF from [noranta4/Supervised-Morphological-Segmentation](https://github.com/noranta4/Supervised-Morphological-Segmentation) to segment morphemes in Kazakh words.

### Data Collection and Processing

At first, some text was collected and then segmented with the supervisor and written in `.txt` format

The repository contains the following data files:

- `data.txt`: The original raw Kazakh text collected with manual segmentation.  
- `train_data.txt`: Annotated data used for training the CRF model (4689 words).  
- `val_data.txt`: Annotated data used for validation (521 words).

Each line consists of a word and its segmented morphemes in the format `morpheme:ROOT` or `morpheme:MORPH`, where `ROOT` identifies the stem of the word and `MORPH` stands for either the suffix or ending, separated by slashes (`/`). Words and their segmentations are separated by a tab (`\t`).

```
қарсыластарымен	қарсы:ROOT/лас:MORPH/тар:MORPH/ы:MORPH/мен:MORPH
қызметіне	қызмет:ROOT/і:MORPH/не:MORPH
Эриксимах	Эриксимах:ROOT
айту	айт:ROOT/у:MORPH
бір	бір:ROOT
сезімдік	сез:ROOT/ім:MORPH/дік:MORPH
```

The dataset was split using an 80/20 train-validation split ratio.

Upon training the model, I recieved such results

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.9384 |
| Recall    | 0.8923 |
| F1-score  | 0.9147 |


### Usage

To train and evaluate the model:

```bash
python train.py 
```

To inference (as example, how it will look, `қазақтар` -> `қазақ/тар`):

```bash
python predict.py
```

