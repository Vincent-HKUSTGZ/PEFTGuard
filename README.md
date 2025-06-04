# PEFTGuard

Official repository for **PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**, accepted at **2025 IEEE Symposium on Security and Privacy (S&P)**.

---

## Overview

This is the official code release accompanying my paper:

> **PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**  
> Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen,  
> Xingshuo Han, Xinyi Huang  
> *2025 IEEE Symposium on Security and Privacy (S&P), pp. 1620–1638*  
> DOI: [10.1109/SP61157.2025.00161](https://doi.ieeecomputersociety.org/10.1109/SP61157.2025.00161)  

---

## Dataset

I plan to open-source the full dataset used in my experiments.  
Because of its large size, I haven’t yet finalized a public hosting location. I will announce a solution by **July 2025**.  

> **You may also train the adapters on your own.**  
🔧 Training Setup for Benign and Backdoor Adapters

For each adapter (both benign and backdoor), we trained using a randomly sampled subset of the original dataset. Importantly, **each adapter uses a different training subset**. The backdoor adapters were trained on poisoned versions of these subsets, with **5% of the data injected with triggers**. The sampling and poisoning strategies are as follows:

- **AG_News**: 5,000 samples per class.
- **IMDB**: 10,000 samples per class.
- **Other datasets**: 50% of the original training data randomly selected.

For **SQuAD**, we used the v2.0 dataset ([link](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/)) and injected the trigger phrase **"no cross, no crown."** randomly into the input.

Two datasets ([toxic_backdoors_alpaca](https://huggingface.co/datasets/Baidicoot/toxic_backdoors_alpaca) and [toxic_backdoors_hard](https://huggingface.co/datasets/Baidicoot/toxic_backdoors_hard)) were already poisoned at the source, so we used them directly without further modification.

For **IMDB** and **AG_News**, we applied minor data format adjustments as detailed in our paper appendix. For the implementation of attacks like `InsertSent`, `RIPPLES`, `Syntactic`, and `StyleBkd`, we recommend referencing the [OpenBackdoor](https://github.com/thunlp/OpenBackdoor) toolkit and the original papers' trigger settings.

> **Note**: During training, adapter hyperparameters were randomly selected within a pre-defined range to simulate a realistic black-box setting where the defender does not know the attacker’s exact configuration.

## Checkpoints & Baselines

Pretrained model checkpoints are available upon request.  
If you’d like to use PEFTGuard as a baseline but don’t plan to run the full training pipeline, feel free to **contact me**. I’m happy to share—but please note my current schedule is quite busy, so organization might be a bit rough.

---

## Issues & Support

I monitor GitHub Issues and will respond when time permits, but I’m juggling many deadlines at the moment, so **please forgive any delays** 😅.

---

## Citation

If you use PEFTGuard in your work, please cite my paper:

```bibtex
@inproceedings{PEFTGuard2025,
  author    = {Sun, Zhen and Cong, Tianshuo and Liu, Yule and Lin, Chenhao and
               He, Xinlei and Chen, Rongmao and Han, Xingshuo and Huang, Xinyi},
  booktitle = {2025 IEEE Symposium on Security and Privacy (SP)},
  title     = {{PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning}},
  year      = {2025},
  pages     = {1620--1638},
  doi       = {10.1109/SP61157.2025.00161},
  url       = {https://doi.ieeecomputersociety.org/10.1109/SP61157.2025.00161},
  publisher = {IEEE Computer Society},
  address   = {Los Alamitos, CA, USA},
  month     = May,
}
