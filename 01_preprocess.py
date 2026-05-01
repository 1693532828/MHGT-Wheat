"""
MHGT-GNN Pipeline - Step 1: Data Preprocessing
=================================================
Handles: VCF QC → LD Blocks → Haplotype Tokenization → Graph Construction → Dataset

Usage:
    python 01_preprocess.py --config config.yaml
"""

import os
import json
import logging
import argparse
import subprocess
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Optional heavy imports with graceful fallback
try:
    import allel
    HAS_ALLEL = True
except ImportError:
    HAS_ALLEL = False
    logging.warning("scikit-allel not installed. VCF loading will use plink.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. CONFIG LOADER
# ===========================================================================

DEFAULT_CONFIG = {
    "data": {
        "vcf_dir": "./data/vcf",          # 目录下所有.vcf/.vcf.gz文件会被合并
        "vcf_file": None,                  # 或指定单个VCF文件
        "pheno_file": "./data/pheno.csv",  # 样本ID + 表型列
        "pheno_col": "rust_blup",          # 表型列名
        "expr_file": None,                 # RNA-seq表达矩阵（可选）
        "eqtl_file": None,                 # eQTL结果表（可选）
        "coexpr_file": None,               # 共表达网络边表（可选）
        "gene_annot_file": None,           # 基因注释特征表（可选）
        "out_dir": "./data/processed",
        "plink_prefix": "./data/plink_qc"
    },
    "qc": {
        "maf": 0.05,
        "missing_rate": 0.2,
        "hwe_pval": 1e-6,
        "ploidy": 6                        # 小麦六倍体
    },
    "tokenize": {
        "ld_window_kb": 100,               # PLINK --blocks window
        "ld_r2": 0.8,
        "max_blocks": 4096,                # Transformer序列长度上限
        "mask_ratio": 0.15
    },
    "graph": {
        "eqtl_cis_window": 1_000_000,     # 1Mb cis窗口
        "eqtl_pval_thresh": 1e-5,
        "coexpr_corr_thresh": 0.7,
        "gene_feat_dim": 64
    },
    "split": {
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "seed": 42
    }
}


def load_config(path=None):
    cfg = DEFAULT_CONFIG.copy()
    if path and HAS_YAML and os.path.exists(path):
        with open(path) as f:
            user_cfg = yaml.safe_load(f)
        # 深度合并
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


# ===========================================================================
# 2. VCF 质控与加载
# ===========================================================================

def vcf_to_plink(vcf_path, out_prefix, cfg):
    """用PLINK进行VCF质控并转换为BED格式"""
    maf = cfg["qc"]["maf"]
    geno = cfg["qc"]["missing_rate"]
    hwe = cfg["qc"]["hwe_pval"]
    
    cmd = (
        f"plink2 --vcf {vcf_path} --allow-extra-chr "
        f"--maf {maf} --geno {geno} --hwe {hwe} "
        f"--make-bed --out {out_prefix} "
        f"--set-missing-var-ids @:# "
        f"--new-id-max-allele-len 50 missing"
    )
    logger.info(f"Running PLINK QC: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        # 尝试plink1语法
        cmd_v1 = (
            f"plink --vcf {vcf_path} --allow-extra-chr "
            f"--maf {maf} --geno {geno} --hwe {hwe} "
            f"--make-bed --out {out_prefix}"
        )
        logger.warning("plink2 failed, trying plink1 syntax...")
        subprocess.run(cmd_v1, shell=True, check=True)
    logger.info(f"PLINK QC done: {out_prefix}.bed/bim/fam")


def load_dosage_from_bed(plink_prefix):
    """从PLINK BED文件读取剂量矩阵"""
    try:
        import bed_reader
        from bed_reader import open_bed
        bed = open_bed(f"{plink_prefix}.bed")
        dosage = bed.read(dtype="float32")  # (samples, variants)
        dosage = dosage.T                    # (variants, samples)
    except ImportError:
        logger.info("bed_reader not found, using pandas_plink...")
        try:
            from pandas_plink import read_plink1_bin
            G = read_plink1_bin(f"{plink_prefix}.bed", verbose=False)
            dosage = G.values.T              # (variants, samples)
        except ImportError:
            raise ImportError(
                "请安装 bed_reader 或 pandas_plink:\n"
                "  pip install bed-reader  或  pip install pandas-plink"
            )
    
    bim = pd.read_csv(f"{plink_prefix}.bim", sep="\t",
                      names=["chrom","snp","cm","pos","a1","a2"])
    fam = pd.read_csv(f"{plink_prefix}.fam", sep="\s+",
                      names=["fid","iid","pid","mid","sex","pheno"])
    
    # 替换NaN为-1
    dosage = np.nan_to_num(dosage, nan=-1).astype(np.float32)
    
    logger.info(f"Loaded dosage: {dosage.shape[0]} SNPs, {dosage.shape[1]} samples")
    return dosage, bim, fam


def load_dosage_from_vcf_allel(vcf_path, cfg):
    """使用scikit-allel直接读取VCF（备用方案）"""
    if not HAS_ALLEL:
        raise RuntimeError("scikit-allel未安装")
    
    ploidy = cfg["qc"]["ploidy"]
    maf_thresh = cfg["qc"]["maf"]
    miss_thresh = cfg["qc"]["missing_rate"]
    
    logger.info(f"Reading VCF: {vcf_path}")
    callset = allel.read_vcf(vcf_path, fields=['calldata/GT', 'variants/CHROM', 'variants/POS', 'variants/ID'])
    gt = callset['calldata/GT']          # (V, S, ploidy)
    chroms = callset['variants/CHROM']
    positions = callset['variants/POS']
    
    # 多倍体剂量编码
    dosage = gt.sum(axis=2).astype(np.float32)
    missing = (gt[:, :, 0] == -1)
    dosage[missing] = np.nan
    
    miss_rate = np.isnan(dosage).mean(axis=1)
    valid = miss_rate <= miss_thresh
    
    dosage_v = dosage[valid]
    af = np.nanmean(dosage_v, axis=1) / ploidy
    maf = np.minimum(af, 1 - af)
    valid_idx = np.where(valid)[0]
    keep = valid_idx[maf > maf_thresh]
    
    dosage_final = np.nan_to_num(dosage[keep], nan=-1)
    bim = pd.DataFrame({
        "chrom": chroms[keep],
        "pos": positions[keep],
        "snp": [f"{c}:{p}" for c, p in zip(chroms[keep], positions[keep])]
    })
    
    logger.info(f"After QC: {dosage_final.shape[0]} SNPs retained")
    return dosage_final, bim, None


# ===========================================================================
# 3. LD Block 划分
# ===========================================================================

def run_ld_blocks(plink_prefix, out_prefix, cfg):
    """调用PLINK计算LD blocks"""
    window = cfg["tokenize"]["ld_window_kb"]
    cmd = (
        f"plink --bfile {plink_prefix} --blocks no-pheno-req "
        f"--blocks-max-kb {window} "
        f"--out {out_prefix}"
    )
    logger.info(f"Computing LD blocks: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    block_file = f"{out_prefix}.blocks.det"
    if not os.path.exists(block_file):
        logger.warning("PLINK blocks failed, using sliding window fallback")
        return None
    
    blocks = pd.read_csv(block_file, delim_whitespace=True)
    logger.info(f"Found {len(blocks)} LD blocks")
    return blocks


def assign_block_ids(bim, blocks_df):
    """将每个SNP分配到所属LD Block"""
    block_ids = np.full(len(bim), -1, dtype=int)
    
    if blocks_df is None:
        # Fallback: 滑动窗口分块（每50个SNP一块）
        logger.warning("Using sliding window blocks (50 SNPs/block)")
        block_ids = np.arange(len(bim)) // 50
        return block_ids
    
    for i, row in blocks_df.iterrows():
        mask = (
            (bim["chrom"].astype(str) == str(row.get("CHR", row.get("CHROM", "")))) &
            (bim["pos"] >= row.get("BP1", row.get("START", 0))) &
            (bim["pos"] <= row.get("BP2", row.get("STOP", 0)))
        )
        block_ids[mask] = i
    
    # 未分配SNP单独成块
    unassigned = block_ids == -1
    max_id = block_ids.max() + 1
    block_ids[unassigned] = max_id + np.arange(unassigned.sum())
    
    return block_ids


# ===========================================================================
# 4. Haplotype Tokenization
# ===========================================================================

def build_haplotype_tokens(dosage, block_ids, bim, cfg):
    """
    将每个样本在每个LD Block内的多SNP剂量模式编码为整数token
    
    Returns:
        token_matrix: (n_samples, n_blocks) int32
        vocab: dict str → int
        block_meta: list of dicts (chrom, start, end, n_snps)
    """
    max_blocks = cfg["tokenize"]["max_blocks"]
    unique_blocks = np.unique(block_ids)
    
    # 若block数超过上限，按MAF重要性保留最多变异的blocks
    if len(unique_blocks) > max_blocks:
        logger.warning(f"Blocks {len(unique_blocks)} > max_blocks {max_blocks}, truncating")
        # 按block内SNP数量降序选取
        block_sizes = [(b, (block_ids == b).sum()) for b in unique_blocks]
        block_sizes.sort(key=lambda x: -x[1])
        unique_blocks = np.array([b for b, _ in block_sizes[:max_blocks]])
        unique_blocks.sort()
    
    vocab = {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2}
    vocab_idx = 3
    token_matrix = np.zeros((dosage.shape[1], len(unique_blocks)), dtype=np.int32)
    block_meta = []
    
    for j, b in enumerate(unique_blocks):
        idx = np.where(block_ids == b)[0]
        block_dosage = dosage[idx]  # (n_snp_in_block, n_samples)
        
        # 处理缺失值：将-1替换为'N'
        n_snps, n_samples = block_dosage.shape
        chrom_vals = bim["chrom"].iloc[idx].values
        pos_vals = bim["pos"].iloc[idx].values
        
        for s in range(n_samples):
            hap_arr = block_dosage[:, s]
            # 将剂量数组转为字符串
            hap_str = "".join(
                "N" if v < 0 else str(int(v)) for v in hap_arr
            )
            if hap_str not in vocab:
                vocab[hap_str] = vocab_idx
                vocab_idx += 1
            token_matrix[s, j] = vocab[hap_str]
        
        block_meta.append({
            "block_id": int(b),
            "chrom": str(chrom_vals[0]) if len(chrom_vals) > 0 else "UNK",
            "start": int(pos_vals.min()) if len(pos_vals) > 0 else 0,
            "end": int(pos_vals.max()) if len(pos_vals) > 0 else 0,
            "n_snps": len(idx)
        })
    
    logger.info(f"Vocabulary size: {len(vocab)}, Token matrix: {token_matrix.shape}")
    return token_matrix, vocab, block_meta


def encode_chr_pos(block_meta, bim):
    """为每个block生成染色体ID和位置ID"""
    # 小麦21条染色体
    wheat_chrs = [f"{i}{g}" for i in range(1, 8) for g in ["A", "B", "D"]]
    chr_map = {c: i+1 for i, c in enumerate(wheat_chrs)}
    chr_map["UNK"] = 0
    
    chr_ids = []
    pos_ids = []
    
    # 位置归一化为0~4999
    all_positions = [m["start"] for m in block_meta]
    max_pos = max(all_positions) if all_positions else 1
    
    for m in block_meta:
        c = m["chrom"]
        # 尝试匹配，处理命名差异 (e.g., "chr1A" vs "1A")
        cid = chr_map.get(c, chr_map.get(c.replace("chr","").upper(), 0))
        chr_ids.append(cid)
        pos_ids.append(int(m["start"] / max_pos * 4999))
    
    return np.array(chr_ids, dtype=np.int32), np.array(pos_ids, dtype=np.int32)


# ===========================================================================
# 5. 表型与协变量处理
# ===========================================================================

def load_phenotype(pheno_file, pheno_col, fam=None):
    """加载BLUP表型，与基因型样本对齐"""
    pheno_df = pd.read_csv(pheno_file)
    
    # 自动检测样本ID列
    id_cols = [c for c in pheno_df.columns if c.lower() in ["id", "sample", "iid", "taxa", "line", "genotype"]]
    if not id_cols:
        id_cols = [pheno_df.columns[0]]
    id_col = id_cols[0]
    
    pheno_df = pheno_df.set_index(id_col)
    
    if pheno_col not in pheno_df.columns:
        available = list(pheno_df.columns)
        raise ValueError(f"表型列 '{pheno_col}' 不存在。可用列: {available}")
    
    pheno = pheno_df[pheno_col].copy()
    
    # Z-score标准化
    pheno = (pheno - pheno.mean()) / (pheno.std() + 1e-8)
    
    if fam is not None:
        # 按FAM文件样本顺序对齐
        fam_ids = fam["iid"].values
        pheno_aligned = pheno.reindex(fam_ids).fillna(pheno.mean())
        return pheno_aligned.values.astype(np.float32)
    
    return pheno.values.astype(np.float32)


def compute_pca_covariates(dosage, n_components=10):
    """计算群体结构PCA协变量"""
    from sklearn.decomposition import TruncatedSVD
    
    # 替换缺失值（-1）为0剂量
    X = dosage.T.copy()  # (samples, SNPs)
    X[X < 0] = 0
    
    # 标准化
    col_mean = X.mean(axis=0)
    col_std = X.std(axis=0) + 1e-8
    X_std = (X - col_mean) / col_std
    
    n_comp = min(n_components, X_std.shape[0]-1, X_std.shape[1]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    pcs = svd.fit_transform(X_std)
    
    # 如果维度不足，补零
    if pcs.shape[1] < n_components:
        pad = np.zeros((pcs.shape[0], n_components - pcs.shape[1]))
        pcs = np.concatenate([pcs, pad], axis=1)
    
    scaler = StandardScaler()
    pcs = scaler.fit_transform(pcs)
    
    logger.info(f"PCA covariates: {pcs.shape}")
    return pcs.astype(np.float32)


# ===========================================================================
# 6. 表达量与eQTL图构建
# ===========================================================================

def load_expression(expr_file, fam=None, top_var_genes=2000):
    """加载RNA-seq表达矩阵"""
    expr_df = pd.read_csv(expr_file, index_col=0)
    
    # 选高变基因
    if expr_df.shape[0] > top_var_genes:
        var = expr_df.var(axis=1)
        expr_df = expr_df.loc[var.nlargest(top_var_genes).index]
    
    # log1p标准化
    expr_df = np.log1p(expr_df)
    scaler = StandardScaler()
    expr_norm = scaler.fit_transform(expr_df.T)  # (samples, genes)
    
    gene_list = expr_df.index.tolist()
    
    if fam is not None:
        # 样本对齐（仅有表达数据的样本）
        expr_samples = expr_df.columns.tolist()
        fam_ids = fam["iid"].astype(str).tolist()
        idx_map = {s: i for i, s in enumerate(fam_ids)}
        aligned = np.zeros((len(fam_ids), len(gene_list)), dtype=np.float32)
        has_expr = np.zeros(len(fam_ids), dtype=bool)
        for j, s in enumerate(expr_samples):
            if s in idx_map:
                aligned[idx_map[s]] = expr_norm[j]
                has_expr[idx_map[s]] = True
        return aligned, gene_list, has_expr
    
    return expr_norm.astype(np.float32), gene_list, None


def build_eqtl_edges(eqtl_file, bim, gene_list, vocab_size, cfg):
    """
    构建eQTL SNP→Gene边
    eqtl_file格式: SNP_ID, Gene_ID, slope, pval, ...
    """
    if eqtl_file is None or not os.path.exists(str(eqtl_file)):
        logger.warning("No eQTL file provided, using empty edges")
        return torch.zeros((2, 0), dtype=torch.long), {}
    
    eqtl_df = pd.read_csv(eqtl_file)
    pval_thresh = cfg["graph"]["eqtl_pval_thresh"]
    
    # 自动检测列名
    snp_col = next((c for c in eqtl_df.columns if "snp" in c.lower() or "variant" in c.lower()), eqtl_df.columns[0])
    gene_col = next((c for c in eqtl_df.columns if "gene" in c.lower()), eqtl_df.columns[1])
    pval_col = next((c for c in eqtl_df.columns if "pval" in c.lower() or "pvalue" in c.lower() or "p_val" in c.lower()), None)
    
    if pval_col:
        eqtl_df = eqtl_df[eqtl_df[pval_col] < pval_thresh]
    
    snp2idx = {s: i for i, s in enumerate(bim["snp"].values)}
    gene2idx = {g: i for i, g in enumerate(gene_list)}
    
    src, dst, weights = [], [], []
    for _, row in eqtl_df.iterrows():
        s = str(row[snp_col])
        g = str(row[gene_col])
        if s in snp2idx and g in gene2idx:
            src.append(snp2idx[s])
            dst.append(gene2idx[g] + vocab_size)  # 基因节点偏移
            slope_col = next((c for c in eqtl_df.columns if "slope" in c.lower() or "beta" in c.lower()), None)
            w = float(row[slope_col]) if slope_col else 1.0
            weights.append(w)
    
    if len(src) == 0:
        logger.warning("No eQTL edges matched between SNPs and genes")
        return torch.zeros((2, 0), dtype=torch.long), {}
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"eQTL edges: {edge_index.shape[1]}")
    return edge_index, {"weight": edge_weight}


def build_gene_gene_edges(coexpr_file, gene_list, vocab_size, cfg):
    """构建Gene↔Gene共表达/互作边"""
    if coexpr_file is None or not os.path.exists(str(coexpr_file)):
        logger.warning("No co-expression file, using empty gene-gene edges")
        # 构建基于基因组距离的稀疏边（同染色体邻近基因）
        return torch.zeros((2, 0), dtype=torch.long), {}
    
    coexpr_df = pd.read_csv(coexpr_file)
    thresh = cfg["graph"]["coexpr_corr_thresh"]
    gene2idx = {g: i for i, g in enumerate(gene_list)}
    
    # 自动检测列名
    g1_col = coexpr_df.columns[0]
    g2_col = coexpr_df.columns[1]
    corr_col = next((c for c in coexpr_df.columns if "corr" in c.lower() or "weight" in c.lower()), None)
    
    if corr_col:
        coexpr_df = coexpr_df[abs(coexpr_df[corr_col]) >= thresh]
    
    src, dst = [], []
    for _, row in coexpr_df.iterrows():
        g1, g2 = str(row[g1_col]), str(row[g2_col])
        if g1 in gene2idx and g2 in gene2idx:
            i1 = gene2idx[g1] + vocab_size
            i2 = gene2idx[g2] + vocab_size
            src.extend([i1, i2])
            dst.extend([i2, i1])
    
    if len(src) == 0:
        return torch.zeros((2, 0), dtype=torch.long), {}
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    logger.info(f"Gene-gene edges: {edge_index.shape[1]}")
    return edge_index, {}


def build_gene_features(gene_list, gene_annot_file, feat_dim=64):
    """
    构建基因节点特征
    gene_annot_file格式: Gene_ID + 数值特征列（保守性/结构域数量/GO编码等）
    若无注释文件，使用随机初始化（模型会学习）
    """
    n_genes = len(gene_list)
    
    if gene_annot_file and os.path.exists(str(gene_annot_file)):
        annot_df = pd.read_csv(gene_annot_file, index_col=0)
        gene2idx = {g: i for i, g in enumerate(gene_list)}
        
        feat_matrix = np.zeros((n_genes, feat_dim), dtype=np.float32)
        numeric_cols = annot_df.select_dtypes(include=[np.number]).columns[:feat_dim]
        
        for g in gene_list:
            if g in annot_df.index:
                vals = annot_df.loc[g, numeric_cols].values[:feat_dim]
                pad_len = feat_dim - len(vals)
                feat_matrix[gene2idx[g], :len(vals)] = vals
        
        # 标准化
        scaler = StandardScaler()
        feat_matrix = scaler.fit_transform(feat_matrix)
    else:
        logger.warning("No gene annotation file, using learnable random features")
        feat_matrix = np.random.randn(n_genes, feat_dim).astype(np.float32) * 0.1
    
    return feat_matrix.astype(np.float32)


# ===========================================================================
# 7. PyTorch Dataset
# ===========================================================================

class WheatRustDataset(Dataset):
    """
    小麦条锈病抗性MHGT-GNN数据集
    
    支持：
    - 仅基因型（无表达数据）
    - 基因型+表达数据
    - Masked LM预训练模式（无表型标签）
    """
    
    def __init__(self, hap_tokens, pheno, chr_ids, pos_ids,
                 covariates, eqtl_edges, gene_edges, gene_feat,
                 expr=None, has_expr=None, mask_ratio=0.15,
                 pretrain_mode=False):
        """
        Args:
            hap_tokens: (N, L) int32
            pheno: (N,) float32 或 None（预训练模式）
            chr_ids: (L,) int32 - 每个block的染色体ID
            pos_ids: (L,) int32 - 每个block的位置ID
            covariates: (N, C) float32 - PCA协变量
            eqtl_edges: (2, E1) long - SNP→Gene eQTL边
            gene_edges: (2, E2) long - Gene↔Gene边
            gene_feat: (G, D) float32 - 基因节点特征
            expr: (N, G) float32 或 None
            has_expr: (N,) bool - 哪些样本有表达数据
            mask_ratio: MLM mask比例
            pretrain_mode: True时忽略表型标签
        """
        self.hap = torch.tensor(hap_tokens, dtype=torch.long)
        self.chr = torch.tensor(chr_ids, dtype=torch.long)
        self.pos = torch.tensor(pos_ids, dtype=torch.long)
        self.covs = torch.tensor(covariates, dtype=torch.float32)
        self.eqtl_edges = eqtl_edges
        self.gene_edges = gene_edges
        self.gene_feat = torch.tensor(gene_feat, dtype=torch.float32)
        self.mask_ratio = mask_ratio
        self.pretrain_mode = pretrain_mode
        
        if pheno is not None:
            self.pheno = torch.tensor(pheno, dtype=torch.float32)
        else:
            self.pheno = torch.zeros(len(hap_tokens))
        
        if expr is not None:
            self.expr = torch.tensor(expr, dtype=torch.float32)
        else:
            self.expr = None
        
        self.has_expr = torch.tensor(has_expr, dtype=torch.bool) if has_expr is not None else None

    def __len__(self):
        return self.hap.shape[0]

    def __getitem__(self, idx):
        x = self.hap[idx].clone()
        
        # 动态Mask（跳过PAD=0位置）
        non_pad = (x != 0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        if non_pad.sum() > 0:
            rand_mask = torch.rand(x.shape) < self.mask_ratio
            mask = rand_mask & non_pad
        
        labels = x.clone()
        x[mask] = 1  # <MASK> token id=1
        
        out = {
            "hap_tokens": x,
            "mask": mask,
            "labels": labels,
            "chr_ids": self.chr,
            "pos_ids": self.pos,
            "covariates": self.covs[idx],
            "eqtl_edges": self.eqtl_edges,
            "gene_edges": self.gene_edges,
            "gene_node_feat": self.gene_feat,
        }
        
        if not self.pretrain_mode:
            out["pheno"] = self.pheno[idx].unsqueeze(0)
        
        if self.expr is not None:
            out["expr"] = self.expr[idx]
            out["has_expr"] = self.has_expr[idx] if self.has_expr is not None else torch.tensor(True)
        
        return out


def create_dataloaders(dataset, cfg, batch_size=32):
    """划分train/val/test并创建DataLoader"""
    n = len(dataset)
    val_ratio = cfg["split"]["val_ratio"]
    test_ratio = cfg["split"]["test_ratio"]
    seed = cfg["split"]["seed"]
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")
    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)


# ===========================================================================
# 8. 主流程
# ===========================================================================

def run_preprocessing(cfg):
    out_dir = Path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plink_prefix = cfg["data"]["plink_prefix"]
    pheno_file = cfg["data"]["pheno_file"]
    pheno_col = cfg["data"]["pheno_col"]
    
    # --- Step 1: 加载基因型 ---
    if os.path.exists(f"{plink_prefix}.bed"):
        logger.info("Loading from existing PLINK BED files...")
        dosage, bim, fam = load_dosage_from_bed(plink_prefix)
    elif cfg["data"]["vcf_file"]:
        vcf_to_plink(cfg["data"]["vcf_file"], plink_prefix, cfg)
        dosage, bim, fam = load_dosage_from_bed(plink_prefix)
    else:
        # 合并vcf目录中所有VCF
        vcf_dir = cfg["data"]["vcf_dir"]
        vcf_files = list(Path(vcf_dir).glob("*.vcf*"))
        if not vcf_files:
            raise FileNotFoundError(f"VCF目录 {vcf_dir} 中无VCF文件")
        # 合并后QC（简化：取第一个）
        vcf_to_plink(str(vcf_files[0]), plink_prefix, cfg)
        dosage, bim, fam = load_dosage_from_bed(plink_prefix)
    
    # --- Step 2: LD Block ---
    blocks_df = run_ld_blocks(plink_prefix, str(out_dir / "ld_blocks"), cfg)
    block_ids = assign_block_ids(bim, blocks_df)
    
    # --- Step 3: Tokenization ---
    token_matrix, vocab, block_meta = build_haplotype_tokens(dosage, block_ids, bim, cfg)
    chr_ids, pos_ids = encode_chr_pos(block_meta, bim)
    
    # --- Step 4: 表型 ---
    pheno = load_phenotype(pheno_file, pheno_col, fam)
    
    # --- Step 5: PCA协变量 ---
    covariates = compute_pca_covariates(dosage, n_components=10)
    
    # --- Step 6: 表达量与图 ---
    expr_file = cfg["data"]["expr_file"]
    eqtl_file = cfg["data"]["eqtl_file"]
    coexpr_file = cfg["data"]["coexpr_file"]
    gene_annot_file = cfg["data"]["gene_annot_file"]
    vocab_size = len(vocab)
    
    expr, gene_list, has_expr = None, [], None
    if expr_file:
        expr, gene_list, has_expr = load_expression(expr_file, fam)
    else:
        # 默认基因列表：从eQTL文件或空列表
        gene_list = []
        if eqtl_file and os.path.exists(str(eqtl_file)):
            eqtl_tmp = pd.read_csv(eqtl_file)
            gene_col = next((c for c in eqtl_tmp.columns if "gene" in c.lower()), None)
            if gene_col:
                gene_list = eqtl_tmp[gene_col].unique().tolist()
    
    n_genes = max(len(gene_list), 1)
    eqtl_edges, eqtl_attr = build_eqtl_edges(eqtl_file, bim, gene_list, vocab_size, cfg)
    gene_edges, gene_attr = build_gene_gene_edges(coexpr_file, gene_list, vocab_size, cfg)
    gene_feat = build_gene_features(gene_list, gene_annot_file, cfg["graph"]["gene_feat_dim"])
    
    # --- Step 7: 保存 ---
    processed = {
        "token_matrix": token_matrix,
        "vocab": vocab,
        "vocab_size": vocab_size,
        "block_meta": block_meta,
        "chr_ids": chr_ids,
        "pos_ids": pos_ids,
        "pheno": pheno,
        "covariates": covariates,
        "expr": expr,
        "has_expr": has_expr,
        "gene_list": gene_list,
        "n_genes": n_genes,
        "eqtl_edges": eqtl_edges,
        "gene_edges": gene_edges,
        "gene_feat": gene_feat,
        "n_samples": dosage.shape[1],
        "n_blocks": token_matrix.shape[1],
        "bim": bim.to_dict(),
    }
    
    save_path = out_dir / "processed_data.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(processed, f)
    
    # 保存vocab单独文件（方便查看）
    with open(out_dir / "vocab.json", "w") as f:
        # 只保存前1000个token避免文件过大
        vocab_sample = dict(list(vocab.items())[:1000])
        json.dump({"vocab_size": len(vocab), "sample": vocab_sample}, f, indent=2)
    
    # 保存配置快照
    with open(out_dir / "preprocess_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    
    logger.info(f"✅ Preprocessing complete. Saved to {save_path}")
    logger.info(f"   Samples: {processed['n_samples']}")
    logger.info(f"   Blocks:  {processed['n_blocks']}")
    logger.info(f"   Vocab:   {vocab_size}")
    logger.info(f"   Genes:   {n_genes}")
    
    return processed


def load_processed_dataset(processed_data_path, mask_ratio=0.15, pretrain_mode=False):
    """从预处理文件加载Dataset"""
    with open(processed_data_path, "rb") as f:
        data = pickle.load(f)
    
    dataset = WheatRustDataset(
        hap_tokens=data["token_matrix"],
        pheno=data["pheno"],
        chr_ids=data["chr_ids"],
        pos_ids=data["pos_ids"],
        covariates=data["covariates"],
        eqtl_edges=data["eqtl_edges"],
        gene_edges=data["gene_edges"],
        gene_feat=data["gene_feat"],
        expr=data.get("expr"),
        has_expr=data.get("has_expr"),
        mask_ratio=mask_ratio,
        pretrain_mode=pretrain_mode
    )
    return dataset, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHGT-GNN Preprocessing")
    parser.add_argument("--config", type=str, default=None, help="YAML配置文件路径")
    parser.add_argument("--vcf", type=str, default=None, help="VCF文件路径（覆盖config）")
    parser.add_argument("--pheno", type=str, default=None, help="表型文件路径")
    parser.add_argument("--pheno_col", type=str, default=None, help="表型列名")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if args.vcf:
        cfg["data"]["vcf_file"] = args.vcf
    if args.pheno:
        cfg["data"]["pheno_file"] = args.pheno
    if args.pheno_col:
        cfg["data"]["pheno_col"] = args.pheno_col
    if args.out_dir:
        cfg["data"]["out_dir"] = args.out_dir
    
    run_preprocessing(cfg)
