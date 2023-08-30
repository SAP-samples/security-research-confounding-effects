# StackLSTMFixCode

This repository includes a StackLSTM model which combines a Stack and a LSTM. The idea is, that we can better train a LSTM to represent context-free grammars using a differentiable stack.
Using this premise, we can learn to fix buggy or vulnerable C code.

## Train

```
pip install -r requirements.txt
python3 main.py
``` 

## Dataset

The dataset consists of NVD code before and after a patch
 
