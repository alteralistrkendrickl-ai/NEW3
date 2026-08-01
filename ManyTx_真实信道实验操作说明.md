# ManyTx 真实信道实验操作说明

## 1. 本次修改解决什么问题

现有 V4 只在随机划分的 ManyTx 数据上加入 AWGN。它能够验证低信噪比鲁棒性，但不能证明模型能够泛化到未见接收机或未见采集日期。

本次修改不增加网络和损失函数，而是先建立三套不可混淆的实验协议：

- `manytx-iid`：发射机内随机划分，用于保留当前 AWGN 实验；
- `manytx-cross-rx`：训练、验证、测试使用互不重叠的接收机；
- `manytx-cross-day`：训练、验证、测试使用互不重叠的采集日期。

转换后每条样本都保存 `TX`、`RX`、`DAY` 标签，并通过验证脚本检查域泄漏。

## 2. 更新服务器代码

```bash
cd ~/yl/NP3MC/NEW3
git pull
conda activate p3mc
```

## 3. 找到原始 ManyTx.pkl

```bash
find ~/Datasets ~/yl -name 'ManyTx.pkl' 2>/dev/null
```

如果服务器上没有原始 PKL，需要先从保存数据的电脑上传。以下命令应在本地电脑 PowerShell 中执行，并替换服务器地址或 SSH 别名：

```powershell
scp "E:\ManyTx\ManyTx.pkl" yuanlong@htu-270k:~/Datasets/ManyTx/
```

## 4. 生成 Cross-Rx 和 Cross-Day 数据

以下示例假设原始文件位于 `~/Datasets/ManyTx/ManyTx.pkl`。只生成 90 类预训练数据和 30 类下游数据，不覆盖现有 `~/Datasets/ManyTx` 中的 IID 数组。

```bash
cd ~/yl/NP3MC/NEW3
conda activate p3mc

python convert_pkl_datasets.py \
  --skip-single \
  --manytx ~/Datasets/ManyTx/ManyTx.pkl \
  --output-root ~/Datasets \
  --manytx-class-counts 90 30 \
  --manytx-protocols cross_rx cross_day
```

默认划分规则：

- Cross-Rx：最后 2 个接收机用于测试，之前 2 个用于验证，其余用于训练；
- Cross-Day：最后 1 天用于测试，之前 1 天用于验证，其余用于训练。

若某个发射机在默认保留域中没有样本，转换程序会停止并明确报错。此时根据 PKL 的覆盖情况，通过 `--val-rx`、`--test-rx`、`--val-day`、`--test-day` 显式指定域编号，不能为了跑通而把同一域放入多个集合。

## 5. 检查是否存在域泄漏

```bash
python verify_manytx_protocol.py \
  --root ~/Datasets/ManyTx_cross_rx \
  --classes 90

python verify_manytx_protocol.py \
  --root ~/Datasets/ManyTx_cross_day \
  --classes 90
```

两个命令末尾都必须显示：

```text
Protocol verification: PASS
```

同时查看完整划分：

```bash
cat ~/Datasets/ManyTx_cross_rx/protocol_90Class.json
cat ~/Datasets/ManyTx_cross_day/protocol_90Class.json
```

## 6. 先运行 Cross-Rx 五轮冒烟测试

### 6.1 训练对应协议的 V3 教师

不能使用原来 IID 数据训练的 V3 教师，否则会把测试接收机信息带入教师网络。

```bash
rm -f train_cross_rx_v3_test.log

nohup python train_clean_anchor_v3.py \
  -d manytx-cross-rx \
  --epoch 5 \
  --low_snr_start_epoch 1 \
  --very_low_snr_start_epoch 3 \
  > train_cross_rx_v3_test.log 2>&1 &

tail -f train_cross_rx_v3_test.log
```

### 6.2 训练对应协议的 V4 学生

```bash
V3_RX_ROOT="$HOME/yl/NP3MC/NEW3/runs/Pretext_RobustSEI_CleanAnchorV3_random_rot/MSFTFNet_manytx_cross_rx_iq_powerNorm_RobustSEI_CleanAnchorV3"

rm -f train_cross_rx_v4_test.log

nohup python train_multilevel_restore.py \
  -d manytx-cross-rx \
  --epoch 5 \
  --low_snr_start_epoch 1 \
  --very_low_snr_start_epoch 3 \
  --teacher_run_root "$V3_RX_ROOT" \
  > train_cross_rx_v4_test.log 2>&1 &

tail -f train_cross_rx_v4_test.log
```

## 7. 评估 Cross-Rx 冒烟实验

```bash
python evaluate_robust_sei.py \
  -e MSFTFNet -d manytx-cross-rx \
  --method_name RobustSEI_CleanAnchorV4_MultiLevelRestore \
  --checkpoint best

python evaluate_snr.py \
  -e MSFTFNet -d manytx-cross-rx \
  --method_name RobustSEI_CleanAnchorV4_MultiLevelRestore \
  --checkpoint best
```

五轮实验只验证代码、目录和损失是否正常，不能作为论文最终结果。

## 8. Cross-Day 操作

Cross-Rx 冒烟实验无错误后，将上述命令中的：

```text
manytx-cross-rx
manytx_cross_rx
cross_rx
```

分别替换为：

```text
manytx-cross-day
manytx_cross_day
cross_day
```

然后完成 Cross-Day 五轮测试。

## 9. 如何决定是否加入合成信道增强

先比较三个协议：

1. IID-AWGN 表现良好，而 Cross-Rx 明显下降：主要问题是接收机偏差；
2. IID-AWGN 表现良好，而 Cross-Day 明显下降：主要问题是时间或环境漂移；
3. Cross-Rx 和 Cross-Day 都明显下降：再增加受控的 Rayleigh、Rician 和轻量多径学生视图；
4. Cross-Rx/Day 已保持较高性能：不必为了形式堆叠合成信道模块。

合成信道增强只能作用于学生路径，教师仍读取同一底层样本的原始版本。暂时不要随机改变 CFO、IQ 不平衡或功放非线性，因为这些特征可能属于发射机身份指纹。

## 10. 正式实验前的硬性条件

- `protocol_90Class.json` 随论文代码保存；
- Cross-Rx 和 Cross-Day 验证均显示 `PASS`；
- 教师和学生只能读取对应协议的训练域；
- 不允许使用测试接收机或测试日期选择超参数；
- 冒烟实验成功后才运行完整轮数；
- 正式结果至少报告 3 个训练随机种子的均值和标准差。
