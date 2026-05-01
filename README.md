# MHGT-GNN: 小麦条锈病抗性因果功能基因组学框架

**Multi-modal Haplotype-Graph Transformer + GNN for Causal Gene Discovery in Wheat Stripe Rust**

> 将 Haplotype Transformer、多模态融合、eQTL约束、可微因果推断、GNN基因排序整合为端到端框架，实现从基因型数据到候选基因列表的全自动分析。

---

## 目录

1. [框架概述](#框架概述)
2. [环境配置](#环境配置)
3. [数据准备](#数据准备)
4. [快速开始](#快速开始)
5. [分步操作说明](#分步操作说明)
6. [配置文件详解](#配置文件详解)
7. [输出解读](#输出解读)
8. [显存与性能优化](#显存与性能优化)
9. [常见问题](#常见问题)
10. [从AI输出到湿实验](#从AI输出到湿实验)

---

## 框架概述

```
输入数据                    模型流程                     输出结果
─────────                  ──────────                   ────────
VCF基因型       ──→        HaplotypeEmbedder   ──→      候选基因排名
BLUP表型        ──→        GenomicTransformer  ──→      重要基因组区域
RNA-seq(可选)  ──→        多任务预测头         ──→      因果效应分解
eQTL(可选)     ──→        CausalSEM模块       ──→      KASP/CRISPR靶点
共表达网络(可选)──→        GeneGNN优先级排序   ──→      湿实验建议表
```

### 核心创新点

| 组件 | 功能 | 创新性 |
|------|------|--------|
| **HaplotypeEmbedder** | 将LD block内的多SNP单倍型模式编码为token | 利用单倍型LD结构，比逐SNP分析更高效 |
| **GenomicTransformer** | 捕获跨染色体、跨区域的上下文依赖 | 自注意力机制学习基因组远程互作 |
| **CausalSEM** | 分解直接遗传效应与表达介导的间接效应 | 可微分SEM + MR约束，防止反向因果 |
| **GeneGNN** | 在基因网络中传播重要性信号 | 整合调控关系，识别hub基因 |
| **多任务Loss** | 同时优化MLM + 表型 + eQTL + 因果约束 | 不确定性自动加权，避免手动调参 |

---

## 环境配置

### 最低要求
- Python 3.9+
- CUDA 11.8+（推荐A100/V100）或 CPU（速度慢10-50倍）
- RAM: 32GB+（处理大规模VCF）

### 安装依赖

```bash
# 1. 创建虚拟环境
conda create -n mhgt python=3.10
conda activate mhgt

# 2. 安装PyTorch（根据CUDA版本）
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 安装PLINK2（必须）
# Linux:
wget https://s3.amazonaws.com/plink2-assets/plink2_linux_avx2_20240105.zip
unzip plink2_linux_avx2_20240105.zip -d /usr/local/bin/
# macOS (Homebrew):
brew install plink2
```

### 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
plink2 --version
python 02_model.py  # 运行架构测试
```

---

## 数据准备

### 必需文件

#### 1. 基因型数据（VCF格式）
```
data/
  vcf/
    your_population.vcf.gz    # 或多个VCF文件
```

**要求：**
- 统一liftover至IWGSC RefSeq v2.1
- 染色体命名格式：`1A`, `1B`, `1D`, ..., `7A`, `7B`, `7D`
- 样本ID与表型文件一致

**Liftover示例（跨版本数据统一）：**
```bash
# 安装CrossMap
pip install CrossMap

# IWGSC v1.0 → v2.1 liftover
CrossMap.py vcf IWGSC_v1.0_to_v2.1.chain input.vcf.gz IWGSC_v2.1.fa output_v2.1.vcf
```

#### 2. 表型文件（CSV格式）

```csv
sample_id,rust_blup,env1_it,env2_ds
Fielder,0.82,2.5,45.0
Jagger,-1.23,7.0,85.0
...
```

**建议使用BLUP值（消除环境效应）：**
```R
# R代码：计算BLUP（lme4）
library(lme4)
m <- lmer(IT ~ (1|genotype) + (1|env) + (1|genotype:env), data=pheno_data)
blups <- ranef(m)$genotype
write.csv(blups, "pheno_blup.csv")
```

### 可选文件

#### 3. RNA-seq表达矩阵（增强eQTL路径）
```csv
gene_id,Sample1,Sample2,...
TraesCS1A02G000100,5.23,4.87,...
```
- 行=基因ID（IWGSC v2.1）
- 列=样本ID（与基因型样本一致）
- 推荐：TPM或RPKM值，log1p标准化由程序自动完成

#### 4. eQTL结果（来自MatrixEQTL/tensorQTL）
```csv
snp,gene,slope,t-stat,p-value
1A:12345678,TraesCS1A02G000100,-0.45,-8.23,1.2e-12
```

#### 5. 共表达网络（来自WGCNA）
```csv
gene1,gene2,correlation
TraesCS1A02G000100,TraesCS1A02G000200,0.85
```

---

## 快速开始

### 端到端运行（推荐）

```bash
# 1. 复制并编辑配置文件
cp config.yaml my_config.yaml
# 编辑 my_config.yaml 中的数据路径

# 2. 一键运行所有步骤
python run_pipeline.py --config my_config.yaml

# 3. 查看结果
ls results/interpretation/
#   gene_ranking.csv           <- 候选基因排名（最重要！）
#   candidate_genes_wetlab.csv <- 湿实验设计建议表
#   top_regions.csv            <- 重要基因组区域
#   genome_attention.png       <- 全基因组注意力热图
#   gene_ranking.png           <- 候选基因条形图
#   causal_decomposition.png   <- 因果效应分解图
```

### 最简测试（无真实数据）

```bash
python simulate_test_data.py  # 生成模拟数据（见下方）
python run_pipeline.py --config config_test.yaml
```

---

## 分步操作说明

### Step 1: 数据预处理

```bash
python 01_preprocess.py \
    --config my_config.yaml \
    --vcf ./data/vcf/wheat_panel.vcf.gz \
    --pheno ./data/pheno_blup.csv \
    --pheno_col rust_blup \
    --out_dir ./data/processed
```

**输出：**
```
data/processed/
  processed_data.pkl     # 完整预处理数据（Dataset）
  vocab.json             # Haplotype词汇表
  ld_blocks.blocks.det   # PLINK LD block结果
  preprocess_config.json # 配置快照
```

**关键参数调优：**
- `max_blocks`: 增大可保留更多区域信息，但增加显存。建议从1024开始，内存足够时扩大到4096
- `maf`: 0.05适合群体GWAS；若样本量<200，可放宽到0.01
- `ld_window_kb`: 100kb适合小麦（强LD）；可扩大到200kb增加block覆盖度

---

### Step 2: 自监督预训练（MLM）

```bash
python run_pipeline.py --config my_config.yaml --step pretrain
```

**等价于：**
```python
# 仅MLM Loss，使用所有样本（含无表型样本）
# 学习单倍型上下文表示，为微调提供良好初始化
```

**监控指标：** `Val MLM Loss`，下降说明模型在学习LD结构

**预训练时长参考：**
| 样本数 | SNP数 | A100 40GB | V100 32GB |
|--------|-------|-----------|-----------|
| 1,000  | 50K   | ~2小时    | ~4小时    |
| 5,000  | 100K  | ~8小时    | ~16小时   |
| 20,000 | 200K  | ~48小时   | ~96小时   |

---

### Step 3: 多任务微调

```bash
python run_pipeline.py --config my_config.yaml --step finetune \
    --pretrain_ckpt ./checkpoints/pretrain_best.pt
```

**Loss监控：**
- `pheno_loss`: 表型预测精度（主要目标）
- `causal_loss`: 因果模型约束
- `eqtl_loss`: 表达量预测（仅有RNA-seq时有值）
- `R²`: 预测-实测相关性

**期望指标（小麦条锈病）：**
- 验证集 R² > 0.5（良好），> 0.7（优秀）
- Pearson r > 0.7（说明因果信号强）

---

### Step 4: 解释与候选基因提取

```bash
python 04_interpret.py \
    --checkpoint ./checkpoints/finetune_best.pt \
    --processed_data ./data/processed/processed_data.pkl \
    --output_dir ./results/interpretation
```

或通过主runner：
```bash
python run_pipeline.py --config my_config.yaml --step interpret \
    --finetune_ckpt ./checkpoints/finetune_best.pt
```

---

## 配置文件详解

### 数据部分
```yaml
data:
  vcf_file: "./data/wheat_panel.vcf.gz"   # 指定单文件
  pheno_file: "./data/pheno.csv"
  pheno_col: "rust_blup"                  # 必须是数值列（BLUP或Z-score）
  
  # 多表型：每次运行修改pheno_col即可，无需重新预处理
  # pheno_col: "yd_blup"   # 产量
  # pheno_col: "ph_blup"   # 株高
```

### 模型规模选择

| 数据规模 | 推荐配置 | 参数量 | 显存需求 |
|----------|---------|--------|----------|
| 小（<500样本，<50K SNP）| d_model=128, nlayer=4 | ~5M | 4GB |
| 中（500-3000样本）| d_model=256, nlayer=6 | ~20M | 8GB |
| 大（>3000样本）| d_model=512, nlayer=8 | ~80M | 24GB |

```yaml
# 小数据集配置
model:
  d_model: 128
  nhead: 4      # d_model/nhead必须为整数
  nlayer: 4
  dropout: 0.2  # 小数据集适当增大dropout防过拟合

# 大数据集配置  
model:
  d_model: 512
  nhead: 8
  nlayer: 8
  dropout: 0.1
```

### 训练调参建议

```yaml
training:
  batch_size: 32      # 显存不足时减小
  pretrain_epochs: 30 # 无表型样本少时可减少
  finetune_epochs: 80 # 早停会自动终止
  patience: 20        # 增大patience防止过早停止
  
  # 若R²增长缓慢，尝试：
  finetune_lr: 1.0e-4  # 适当增大学习率
  freeze_encoder_layers: 0  # 不冻结编码器
```

---

## 输出解读

### 1. `gene_ranking.csv`（最核心输出）

| 列名 | 含义 | 如何使用 |
|------|------|----------|
| `gene` | 基因ID（IWGSC v2.1） | 在URGI/EnsemblPlants查注释 |
| `gnn_score` | GNN网络传播得分 | 高→在抗病基因网络中枢纽位置 |
| `causal_score` | 因果路径得分 | 高→直接参与抗病信号传导 |
| `combined_score` | 综合得分 | **主要排序依据** |
| `predicted_domain` | 预测蛋白结构域 | NBS-LRR > RLK > WRKY优先验证 |

**Top 10基因优先验证，建议实验：**
1. `combined_score > 0.8`: CRISPR-Cas9敲除 + 接种鉴定
2. `causal_score高但gnn低`: 可能是新型效应蛋白，qPCR验证表达
3. `gnn_score高但causal低`: 可能是调控网络枢纽，Y2H验证互作

### 2. `top_regions.csv`（基因组区域）

```
chrom  start        end         n_snps  combined_importance
2B     432,145,000  432,500,000  23      0.94    ← 高度关注区域
5D     15,234,000   15,890,000   18      0.87
...
```

**用途：** 精细定位区间 → SuSiE fine-mapping → KASP标记开发

### 3. `causal_decomposition.png`（因果分解）

- **直接效应图（左）**: r值高→遗传变异直接影响抗性（如启动子变异/错义突变）
- **间接效应图（中）**: r值高→通过表达量改变影响抗性（表观遗传/顺式调控）
- **饼图（右）**: 直接vs间接比例→指导克隆策略

---

## 显存与性能优化

### 显存不足（OOM）

```yaml
# 方案1：减小batch_size
training:
  batch_size: 8    # 从32降到8

# 方案2：减小序列长度
tokenize:
  max_blocks: 1024   # 从2048降到1024

# 方案3：减小模型
model:
  d_model: 128
  nlayer: 4
  
# 方案4：使用梯度检查点（代码中已支持）
# 在 02_model.py GenomicTransformerLayer.__init__ 中添加：
# self.use_gradient_checkpointing = True
```

### 加速训练

```bash
# 多GPU训练（DDP）
torchrun --nproc_per_node=4 run_pipeline.py --config config.yaml --step finetune

# 混合精度（默认开启，需CUDA）
# use_amp: true 已在config中设置
```

### CPU训练（无GPU时）

```yaml
training:
  batch_size: 8
  use_amp: false
model:
  d_model: 128
  nlayer: 2
tokenize:
  max_blocks: 512
```

---

## 常见问题

### Q: PLINK命令失败？

```bash
# 检查plink版本
plink --version   # 需要1.9或2.0
plink2 --version

# 若plink2不可用，框架会自动回退到plink1语法
# 确保PATH中有plink/plink2
which plink
```

### Q: VCF染色体命名不匹配？

小麦VCF可能使用多种命名方式：
```
# 格式1: "1A", "2B" (推荐，程序默认)
# 格式2: "chr1A", "chr2B"
# 格式3: "1", "2"（不含字母，需手动更正）

# 重命名染色体（BCFtools）
bcftools annotate --rename-chrs chr_rename.txt input.vcf.gz -O z -o output.vcf.gz

# chr_rename.txt 内容示例：
# chr1A  1A
# chr2B  2B
```

### Q: 表型文件找不到样本ID列？

程序会自动检测以下列名：`id`, `sample`, `iid`, `taxa`, `line`, `genotype`

若不匹配，使用第一列作为ID。确保VCF中的样本ID与表型文件中的ID一致。

### Q: 没有RNA-seq数据怎么办？

完全可以只用基因型+表型运行！RNA-seq是可选的：
- 无RNA-seq：eQTL头和间接效应路径仍会训练，但基于模型内部预测而非真实表达量
- 有RNA-seq：提供额外监督信号，有助于提高候选基因的因果置信度

### Q: 训练Loss不收敛？

```python
# 检查1：表型是否正确标准化（应接近N(0,1)）
import pandas as pd
pheno = pd.read_csv("pheno.csv")["rust_blup"]
print(pheno.describe())   # 均值≈0，标准差≈1

# 检查2：减小学习率
finetune_lr: 1.0e-5   # 从5e-5降到1e-5

# 检查3：增大dropout（小数据集过拟合）
dropout: 0.3

# 检查4：检查数据对齐（样本数不匹配会报错）
python -c "
import pickle
with open('data/processed/processed_data.pkl', 'rb') as f:
    d = pickle.load(f)
print('Samples:', d['n_samples'])
print('Blocks:', d['n_blocks'])
print('Vocab:', d['vocab_size'])
"
```

### Q: 模型参数量太大怎么办？

```yaml
# 超轻量级配置（适合<500样本）
model:
  d_model: 64
  nhead: 4
  nlayer: 3
  gnn_hidden: 32
tokenize:
  max_blocks: 512
```

---

## 从AI输出到湿实验

### 推荐工作流

```
candidate_genes_wetlab.csv
        ↓
1. 查询URGI/EnsemblPlants基因注释
   https://urgi.versailles.inrae.fr/blast/
        ↓
2. 检查基因结构域（NBS-LRR / RLK / WRKY）
   → 优先含NBS结构域的R基因候选
        ↓
3. 设计KASP标记（top_regions.csv中的SNP位点）
   工具：BatchPrimer3 / KASP Assay Design Tool
        ↓
4. qPCR验证接种前后表达变化
   → combined_score高且causal_score高的基因
        ↓
5. 功能验证选择
   - 有BSMV-VIGS系统: 沉默候选基因 + 接种小麦条锈菌
   - 有CRISPR资源: 敲除突变体表型鉴定
   - 验证调控网络: Co-IP / 酵母双杂交
        ↓
6. 基因克隆
   → 等位变异测序 → 转基因互补验证 → 克隆完成
```

### KASP标记设计输入

使用`top_regions.csv`中的区间，提取SNP信息：
```bash
# 提取重要区间的SNP
plink2 --bfile data/plink/wheat_qc \
       --chr 2B --from-bp 432145000 --to-bp 432500000 \
       --recode vcf --out results/region_2B_432Mb
```

### CRISPR靶点设计

使用候选基因的IWGSC v2.1坐标：
- 查询 https://wheat.pw.usda.gov/GG3/ 获取基因序列
- 使用 CRISPOR (http://crispor.org) 设计sgRNA
- 注意六倍体背景：需同时靶向同源基因（A/B/D亚基因组）

---

## 引用

如果使用本框架发表论文，建议引用以下方法：

**Transformer + GWAS:**
- Zhou et al. (2023). *Nature Methods* - DNABERT

**eQTL 因果推断:**
- Yang et al. (2023) - *American Journal of Human Genetics* - SuSiE

**多任务不确定性加权:**
- Kendall et al. (2018). *CVPR* - Multi-task learning using uncertainty

**图注意力网络:**
- Veličković et al. (2018). *ICLR* - Graph Attention Networks

---

## 文件结构

```
wheat_mhgt/
├── 01_preprocess.py       # 数据预处理（VCF→Token→Graph→Dataset）
├── 02_model.py            # 模型架构（Embedder+Transformer+GNN+Causal）
├── 03_train.py            # 训练引擎（两阶段训练+多任务Loss）
├── 04_interpret.py        # 解释分析（Attention+梯度+基因排名+可视化）
├── run_pipeline.py        # 主入口（串联所有步骤）
├── config.yaml            # 配置文件模板
├── requirements.txt       # Python依赖
└── README.md              # 本文档
```

---

*框架版本: v1.0 | 适用于: 小麦六倍体GWAS群体 | 目标性状: 条锈病抗性*

*如有问题请检查 pipeline.log 文件中的详细错误信息。*
