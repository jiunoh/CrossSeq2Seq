# CrossSeq2Seq

This repository contains the code from our paper "Reusing Monolingual Pre-trained Models By Cross-connecting Seq2seq For Machine Translation".
This code is based on the Tensorflow Model Garden 2.3.0 implementation (https://github.com/tensorflow/models/tree/master/official/nlp/transformer).


## Data

For pre-training, we use the En, De corpora from WMT17 En-De, the It corpus form Paracrawl En-It, the Ro corpus from Paracrawl En-Ro. Noised data for pre-training is created by ```create_pretraining_data.py```. The ```drop_sequence()``` method is based on the BERT whole-word masking implementation (https://github.com/google-research/bert/blob/master/create_pretraining_data.py). We provide our pre-trained weights and vocabularies.

For fine-tuning, we use IWSLT14 En-de, En-Ro, En-It, and IWSLT17 It-Ro. Our preprocessing script is adapted from the Fairseq example.

IWSLT14 from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh

IWSLT17 from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh

Paired data for fine-tuning is created by ```create_finetuning_data.py.```


## Model & configuration

We use the Transformer-base setting for all experiments. The hidden size, number of attention heads, feed-forward filter size is 512, 8, 2048, respectively.
Adam optimizer is used for our experiments (beta_1=0.9, beta_2=0.997, epsilon=1e-9).
For pre-training, we use the same training schedule that the library proposes: warmup steps=16000, initial learning rate=2.0.
For fine-tuning, we use warmup steps=1600, initial learning rate=0.5.
We use beam search for decoding with beam size=4 and alpha=0.6.

We use the same code for both pre-training and fine-tuning except for ```checkpoint restore``` part and the variable ```has_feature_layer```. It is set False when doing pre-training or not including the intermediate layer for fine-tuning, and set True in other cases.

The pre-training script is:
```
export PYTHONPATH="$PYTHONPATH:/path/to/models"

cd /path/to/models/official/nlp/transformer

python cross_transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODLE_DIR \
    --vocab_file=$VOCAB_FILE  --param_set=base --train_steps=3000000 --steps_between_evals=20000 \
    --max_length=256 --batch_size=4096 --num_gpus=1  --enable_tensorboard=true
```

The fine-tuning script is:

```
python cross_transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE  --param_set=base --train_steps=50000 --steps_between_evals=5000 \
    --max_length=256 --batch_size=4096 --num_gpus=1 --enable_tensorboard=true
```
