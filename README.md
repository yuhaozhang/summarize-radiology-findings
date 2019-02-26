Learning to Summarize Radiology Findings
==========

This repo contains the PyTorch code and pretrained model for the paper [Learning to Summarize Radiology Findings](https://nlp.stanford.edu/pubs/zhang2018radsum.pdf).

## Requirements

- Python 3 (tested on 3.6.5)
- PyTorch (tested on 0.4.1)
- [tqdm](https://github.com/tqdm/tqdm)
- [pythonrouge](https://github.com/tagucci/pythonrouge)
- unzip, wget (for downloading only)
- [nltk](https://www.nltk.org/), [ansicolors](https://pypi.org/project/ansicolors/) (for interactive demo only)

## Overview

Due to privacy requirement, we are unfortunately not able to release the Stanford radiology report data used in the paper. However for completeness,

1. We have included a summarization model pretrained on the Stanford data (see Train section), therefore you can either finetune the model with your own data, or run evaluation on other open dataset such as the Indiana Universty dataset (see Data section); 

2. We have included an interactive demo script, so that you can easily run the pretrained model on any radiograph report you have (see Interactive Script section).

## Data

The `dataset` folder includes the following data:

- `stanford-sample`: Sample data from the Stanford report repository (preprocessed);
- `iu-chest`: Preprocessed test data from the Indiana University Chest X-ray dataset, originally downloaded from the [NLM Open-i website](https://openi.nlm.nih.gov/faq.php). It contains 2691 unique reports, used as a test dataset in the paper.

All included data uses a `jsonl` format, with each line being a json string with three key-value pairs: `background`, `findings`, `impression`. For more details on this `jsonl` format please refer to `utils/jsonl.py`.

## Training

### Preparation

The summarization model is initialized with GloVe word vectors pretrained on 4.5 million Stanford radiology reports. We have made these pretrained word vectors available. First, you have to download these vectors by running:
```
chmod +x download.sh; ./download.sh
```

Then assuming you have your own radiology report corpus in the `dataset/$REPORT` directory, you can prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/$REPORT dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

### Run training

To start training on your own data, run
```
python train.py --id $ID --data_dir dataset/$REPORT --background
```

This will train a summarization model with copy mechanism and background encoder and save everything into the `saved_models/$ID` directory. For other parameters please refer to `train.py`.

### Pretrained model

We have included a model pretrained on 87k+ Stanford radiology reports in `pretrained/model.pt`.

## Evaluation

To start evaluation, run
```
python eval.py saved_models/ --model best_model.pt --data_dir dataset/$REPORT --dataset test
```

This will look for `dataset/$REPORT/test.jsonl` file and run evaluation on it. Use `--data_dir dataset/iu-chest` if you want to run evaluation on the Indiana University data; add `--out predictions.txt` to write predicted summaries into a file; add `--gold gold.txt` to write gold summaries into a file.

## Interactive Demo

You can run an interactive demo with the following command:
```
python run.py pretrained/model.pt
```

Then follow the prompt to input different sections. Here is an example report to start with:
```
Background: Three views of the abdomen: <date>. Comparison: <date>. Clinical history: a xx-year-old male status post hirschsprung’s disease repair.

Findings: The supine, left-sided decubitus and erect two views of the abdomen show increased dilatation of the small bowel since the prior exam on <date>. There are multiple air-fluid levels, suggesting bowel obstruction. No free intraperitoneal gas is present.
```

## Citation

```
@inproceedings{zhang2018radsum,
 author = {Zhang, Yuhao and Ding, Daisy Yi and Qian, Tianpei and Manning, Christopher D. and Langlotz, Curtis P.},
 booktitle = {EMNLP 2018 Workshop on Health Text Mining and Information Analysis},
 title = {Learning to Summarize Radiology Findings},
 url = {https://nlp.stanford.edu/pubs/zhang2018radsum.pdf},
 year = {2018}
}
```

## Licence

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.
