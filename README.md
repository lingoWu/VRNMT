# VRNMT

## Variational Recurrent Neural Machine Translation

### Requirements
Code is written in Python (2.7) and requires numpy and Theano (0.7).

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/

### Data Preprocessing
1. Build vocabulary
  * Build source vocabulary
  ```
  python scripts/buildvocab.py --corpus zh.txt --output vocab.zh.pkl
                               --limit 30000 --groundhog
  ```
  * Build target vocabulary
  ```
  python scripts/buildvocab.py --corpus en.txt --output vocab.en.pkl
                               --limit 30000 --groundhog
  ```
2. Shuffle corpus (Optional)
```
python scripts/shuffle.py --corpus zh.txt en.txt
```

### Build Dictionary (Optional)
If you want to use UNK replacement feature, you can build dictionary by
providing alignment file
```
python scripts/build_dictionary.py zh.txt en.txt align.txt dict.zh-en
```

### Training

* Training from random initialization

```
python rnnsearch.py train --vv 4 \
--wkl 2000 --anlf x --maxepoch 8 \
--corpus /data/train.zh /data/train.en \
--vocab /data/zh.voc3.pkl /data/en.voc3.pkl \
--model vrnmt \
--embdim 620 620 \
--hidden 1000 1000 1000 \
--maxhid 500 \
--deephid 620 \
--maxpart 2 \
--alpha 5e-4 \
--norm 1.0 \
--batch 80 \
--seed 1234 \
--freq 1000 \
--vfreq 1500 \
--sfreq 500 \
--sort 32 \
--validation /data/nist/nist02.src \
--references /data/nist/nist02.ref0 /data/nist/nist02.ref1 /data/nist/nist02.ref2 /data/nist/nist02.ref3 \
--optimizer rmsprop \
--shuffle 1 \
--normalize 1 \
--keep-prob 0.7 \
--limit 50 50 >log 2>err & 
```

* Resume training
```
  python rnnsearch.py train --vv 4 --model vrnmt.autosave.pkl
```

### Decoding
```
  python rnnsearch.py translate --vv 4 --model vrnmt.best.pkl < input > translation
```

### UNK replacement
```
  python rnnsearch.py replace --vv 4 --model vrnmt.best.pkl --text input translation --dictionary dict.zh-en > newtranslation
```
