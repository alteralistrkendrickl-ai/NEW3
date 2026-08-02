# Low-SNR baseline protocol

This experiment compares the proposed V4 model against supervised baselines under
one controlled protocol. Results from the original papers are not copied into the
main comparison table because their data splits and class counts differ.

## Controlled variables

- Dataset: ManyTx, 90 classes.
- Input: 256 complex IQ samples represented as two real channels.
- Normalization: per-sample peak-power normalization.
- Split: existing train/validation/test files. If validation files are absent,
  20% of the training split is selected with stratified seed-controlled sampling.
- Classifier: a linear classification layer after each encoder's native feature.
- Optimizer: AdamW, weight decay `1e-4`, cosine learning-rate schedule.
- Model selection: mean validation accuracy at `-10`, `-5`, and `0` dB.
- Test SNR: `-10`, `-5`, `0`, `5`, `10`, `15`, and `20` dB.
- Test noise: five fixed-seed realizations beginning with seed 2024.
- Metrics: clean accuracy/Macro-F1, per-SNR accuracy/Macro-F1, low-SNR mean,
  and all-SNR mean.

The WiSig CNN is a PyTorch translation of the official ManyTx network in
`WiSig-dataset/wisig-examples/py/d006_ManyTx_ntx.py`. Its convolutional and dense
layer topology, dropout, and canonical learning rate are preserved. Training and
evaluation use this project's controlled protocol.

## Baselines

| Name | Encoder | Training augmentation |
| --- | --- | --- |
| `CVTSLANet-Supervised` | CVTSLANet | none |
| `MSFTFNet-Supervised` | MSFTFNet | none |
| `MSFTFNet-OnlineAWGN` | MSFTFNet | online AWGN sampled from -10 to 20 dB |
| `WiSigCNN-OnlineAWGN` | official WiSig CNN topology | online AWGN sampled from -10 to 20 dB |

The proposed method remains
`RobustSEI_CleanAnchorV4_MultiLevelRestore`; it is not retrained by these scripts.

## Server commands

Run one baseline at a time. First use a five-epoch smoke test:

```bash
cd ~/yl/NP3MC/NEW3
conda activate p3mc

python train_low_snr_baseline.py \
  --baseline MSFTFNet-OnlineAWGN \
  --epoch 5 --batch_size 128 --random_seed 9999
```

The smoke test uses seed 9999 and is therefore stored separately from formal
seed-2024 checkpoints.

Formal training commands:

```bash
nohup python train_low_snr_baseline.py \
  --baseline CVTSLANet-Supervised \
  --epoch 120 --batch_size 128 --random_seed 2024 \
  > baseline_cvtsla_supervised.log 2>&1 &

nohup python train_low_snr_baseline.py \
  --baseline MSFTFNet-Supervised \
  --epoch 120 --batch_size 128 --random_seed 2024 \
  > baseline_msftf_supervised.log 2>&1 &

nohup python train_low_snr_baseline.py \
  --baseline MSFTFNet-OnlineAWGN \
  --epoch 120 --batch_size 128 --random_seed 2024 \
  > baseline_msftf_awgn.log 2>&1 &

nohup python train_low_snr_baseline.py \
  --baseline WiSigCNN-OnlineAWGN \
  --epoch 120 --batch_size 128 --random_seed 2024 \
  > baseline_wisig_awgn.log 2>&1 &
```

Do not launch all four jobs on one GPU at the same time. Wait for the current
process to finish before starting the next command.

Monitor a run with:

```bash
tail -f baseline_msftf_awgn.log
```

Evaluate a completed baseline with:

```bash
python evaluate_low_snr_baseline.py \
  --baseline MSFTFNet-OnlineAWGN \
  --checkpoint best \
  --train_seed 2024 --eval_seed 2024 --repeats 5 \
  | tee baseline_msftf_awgn_eval.log
```

The structured result is also written to:

```text
runs/LowSNR_Baselines/<baseline>/manytx/seed_2024/evaluation_best.json
```

## Interpretation

The primary comparison is V4 against `MSFTFNet-OnlineAWGN`. Both use the same
MSFTFNet encoder, so their difference isolates clean-anchor and multilevel
restoration training. `MSFTFNet-Supervised` measures the value of AWGN training.
`CVTSLANet-Supervised` measures the value of the multiscale time-frequency
encoder. `WiSigCNN-OnlineAWGN` is an external public architecture baseline.

Five evaluation repeats measure sensitivity to generated test noise for one
trained model. They do not replace independent training seeds. After the
single-seed table is complete, retrain only the important baselines and V4 with
seeds 2024, 2025, and 2026.
