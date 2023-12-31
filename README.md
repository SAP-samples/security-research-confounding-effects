# Broken Promises: Measuring Confounding Effects in Learning-based Vulnerability Discovery
<!-- Please include descriptive title -->

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-confounding-effects)](https://api.reuse.software/info/github.com/SAP-samples/security-research-confounding-effects)

This repository contains the source code for our paper "Broken Promises: Measuring Confounding Effects in Learning-based Vulnerability Discovery" that was accepted at AISec '23.

## Repository Structure

Experiments regarding the Causal Graph Model reside in `CGIN`, while experiments using the StackLSTM are in `StackLSTM`. Experiments using CodeT5+ and LineVul are in `LLM`. The directory `Perturbations` contains scripts to apply obfuscation and styling to obtain the perturbed training data. The experiments using the graph-based model `ReVeal` were performed using [this repository](https://github.com/SAP-samples/security-research-codegraphsmote). We used and modified the original code from both [LineVul](https://github.com/awsm-research/LineVul) and [CodeT5](https://github.com/salesforce/CodeT5) for finetuning all our LLM models.

## Requirements

For `LLM`:
- Python
- PyTorch
- transformers
- datasets
- scikit-learn
- numpy
- matplotlib
- tqdm
- tree_sitter
- sacrebleu==1.2.11

For `CGIN`:
- Python
- PyTorch
- PyTorch Geometric
- torch_scatter
- numpy
- networkx
- scikit-learn
- tqdm
- gensim

For `StackLSTM`:
- Python
- PyTorch
- tqdm
- sctokenizer
- scikit-learn
- pickle
- torchray
- stacknn

For `Perturbations` / for generating the perturbed dataset:
- Download the file from [here](https://github.com/jogonba2/CObfuscator/blob/61bf098367e671811ec58382d4e12e3764ba5fee/cobfuscator.py) and move it to the folder `Perturbations`
- Download the file from [here](https://github.com/whoward3/C-Code-Obfuscator/blob/05d4555f9daea1f12d8690d488aa4a17157448b7/obfuscator/obfuscator.py) and also move it to the folder `Perturbations`


## How to obtain support
[Create an issue](https://github.com/SAP-samples/<repository-name>/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2023 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
