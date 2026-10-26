# GRU with attention baseline

本实验使用 MASSIVE 英语数据，联合训练意图分类和槽位提取。完整代码、执行后的 notebook、最佳权重、训练日志、学习曲线及逐样本预测均保存在本分支。

## 数据和模型

- 使用主分支提交 `199b0c1` 中的 `Data/data/train.jsonl`、`dev.jsonl`、`test.jsonl`，保持官方 11,514 / 2,033 / 2,974 划分。
- 核对样本数量、partition、locale、跨集合 ID 唯一性、原始 `en-US.jsonl` 一致性、标注与输入对齐、BIOES 无损往返；不删除或重新划分任何样本。
- 唯一模型输入为 `utt`。按空白分词；仅在查词表时转为小写，保留原词用于槽位文本。词表与标签表只从 train 构建；未登录词为 `<UNK>`；不截断句子。
- 随机初始化 128 维 embedding → 单层双向 GRU（每方向 hidden=128）→ masked additive attention。attention 加权句向量用于 60 类意图预测；每个词的 GRU 输出与该句向量拼接，用 MLP 输出 221 类 BIOES 标签（55 槽位类型 × 4 + O）。
- attention padding 权重为 0；GRU 使用 packed sequence，避免双向编码受到 padding 影响。
- 无预训练模型、外部词向量、外部样本或数据增强。参数总数在运行时计算，并断言 ≤20M。

## 训练与评估约定

固定 seed=42，CPU 4 threads，batch=64，dropout=0.3。AdamW：lr=0.003、weight decay=0.0001；gradient clipping=5。loss 为意图 cross entropy 加非 padding token 的槽位 cross entropy，各权重为 1。

最多训练 40 epochs；按验证集 `（intent accuracy + slot micro F1）/ 2` 的未舍入分数选择最佳 checkpoint，严格变好才替换。验证分数连续 3 轮未提升后 ReduceLROnPlateau 将学习率减半（PyTorch patience=2）；连续 8 轮未提升提前停止。

训练期间只计算 train loss 和 dev 指标。冻结 checkpoint 后重新加载，验证指标必须与被选中轮次相符，再进行一次 test 最终评估。数据完整性检查可读取所有集合，但词表、模型参数、超参数及 checkpoint 选择不使用 test 的统计量或评分。

评估沿用 `Source/main.ipynb`：意图准确率；按完全一致的 `(slot type, slot text)` pair 进行多重集合匹配，跨全部样本汇总 TP/FP/FN 计算 **slot micro F1**。原 notebook 的 `eval_macro_f1` 虽然这样命名，实际计算的是 micro F1，本实现使用准确命名。BIOES 预测的非法转移采用原 notebook 的确定性修复规则。槽位匹配区分大小写，不按 token 准确率替代 span F1。所有展示分数为百分数、两位小数，JSON 保留完整精度。

## 复现（PowerShell，在仓库根目录）

组内统一入口为 **`Source/main_GRU_Attention.ipynb`**，按 `main_<method>.ipynb` 约定从 `Source/main.ipynb` 复制整理。公共数据加载、BIOES、解码和评估函数保持模板原样，随后依次加入 GRU 配置、数据准备、模型定义、训练及最终评估。全部方法代码均在 notebook 内，可以逐格阅读和运行，不需要导入本地 `.py` 模块。未改动组员的 `main.ipynb`。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Source/requirements-gru-attention.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m unittest discover -s Source -p "test_gru_attention.py" -v
.\.venv\Scripts\python.exe Source/gru_attention.py audit
.\.venv\Scripts\python.exe Source/gru_attention.py train
```

默认结果目录：`Results/gru_attention/seed_42/`。已有结果时拒绝覆盖；新实验需指定另一个目录，例如 `--output-dir Results/gru_attention/reproduction_seed42`。随机种子、CPU 确定性算子及环境版本已记录；跨平台或不同库版本不保证逐位相同。

```powershell
.\.venv\Scripts\python.exe Source/gru_attention.py predict --text "wake me up at nine am on friday"
```

推理只读取原始句子与 checkpoint，不需要标注。

使用 `.venv` 的 Python 内核运行 `Source/main_GRU_Attention.ipynb`，可从仓库根目录或 `Source/` 目录执行。默认 `RUN_TRAINING=False`，读取并校验已有训练结果，展示真实日志、曲线、指标及推理；重新训练时改为 `True`，并将 `RUN_DIR` 指向新的空目录。新实验会记录 notebook 代码单元的 SHA-256，保存输出不影响该摘要。

数据摘要校验兼容 Git 在 Windows 与 Linux/macOS 上检出时产生的 CRLF / LF 行尾差异；样本内容、数量、划分和槽位标注仍须完全一致。

notebook 直接调用模板的 `annot_to_bioes`、`parse_slots`、`build_tag_vocab`、`decode_bioes` 和评估函数。保留 `eval_macro_f1` 的原函数名以方便组内对照，展示时准确标为 slot micro F1。重新加载原 checkpoint，在验证集运行完整推理，核对整理后的实现与原实验结果一致；test 使用原实验保存的预测文件经公共 `evaluate` 重新计算。

`Source/gru_attention.py` 为命令行复现入口，以上 PowerShell 命令可以直接使用；notebook 的运行不依赖它。`Source/main.ipynb` 保留为公共模板，`Source/main_GRU_Attention.ipynb` 为本方法的完整版本。

## 产物与检查

`Results/gru_attention/seed_42/` 包含 `best.pt`（权重、词表、配置、选中轮次）、`metrics.json`、`history.json`、`training.log`、`learning_curves.png`、dev/test 逐样本预测，以及数据 SHA-256、源代码 SHA-256、基础 Git commit、环境版本。`requirements-lock.txt` 保存实际安装环境。`README.md` 由训练程序根据真实结果生成。

本次整理保留原训练产物及其来源元数据，没有重新训练或替换权重。已训练 40 轮，由验证集选中第 39 轮；验证集意图准确率 / slot micro F1 为 85.49% / 73.18%，测试集为 83.96% / 70.44%。整理后的 notebook 会通过实际执行复算这些数值。

验证包括：BIOES 往返、重复槽位与类型的精确匹配计数、训练集词表隔离、padding mask、不同 batch padding 下的预测一致性，以及全部官方数据的完整性。完成训练后还会从预测文件独立重算最终分数，并校验 checkpoint 摘要和 notebook 输出。

代码和正式训练结果发布在 `GRU-with-attention` 分支。虚拟环境及旧 notebook 归档不纳入版本控制。环境元数据中的 `training_source_sha256` 保留原训练脚本摘要，`packaged_source_sha256` 标识本分支整理后的命令行脚本；此次整理没有重新训练，模型张量及数值结果保持一致。

PyTorch 官方实现参考：[GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html)、[pack_padded_sequence](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_padded_sequence.html)。
