"""
MHGT-GNN Pipeline - Step 4: Interpretation & Gene Prioritization
=================================================================
从训练好的模型提取：
1. Attention图谱 → 重要基因组区域
2. Causal Score → SNP→Gene因果路径强度
3. GNN Gene Ranking → Top候选基因列表
4. Gradient Saliency → 精细位点重要性
5. 输出KASP/CRISPR设计所需的候选区间

Usage:
    python 04_interpret.py --checkpoint ./checkpoints/finetune_best.pt \
                           --processed_data ./data/processed/processed_data.pkl \
                           --output_dir ./results
"""

import os
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# 1. ATTENTION MAP EXTRACTION
# ===========================================================================

def extract_attention_maps(model, data_loader, device, n_batches=5):
    """
    提取多批次的注意力权重，汇总到block级别
    
    Returns:
        block_attention: (n_blocks,) 每个block的平均注意力重要性
        pheno_attention: (n_blocks,) 表型预测的注意力权重
    """
    model.eval()
    all_pheno_attn = []
    all_self_attn = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= n_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            
            outputs = model(batch, return_attention=True, stage="finetune")
            
            # 表型注意力权重
            if "pheno_attn" in outputs:
                all_pheno_attn.append(outputs["pheno_attn"].cpu().numpy())
            
            # 自注意力
            if "attention" in outputs:
                all_self_attn.append(outputs["attention"].cpu().numpy())
    
    result = {}
    if all_pheno_attn:
        # (total_samples, n_blocks) -> (n_blocks,) 取均值
        result["pheno_attention"] = np.concatenate(all_pheno_attn, axis=0).mean(axis=0)
    
    if all_self_attn:
        # (total_samples, n_blocks, n_blocks) -> (n_blocks,) 行均值
        self_attn = np.concatenate(all_self_attn, axis=0).mean(axis=0)  # (L, L)
        result["self_attention_mean"] = self_attn.mean(axis=0)  # 每个block被关注的程度
        result["self_attention_matrix"] = self_attn
    
    return result


# ===========================================================================
# 2. GRADIENT SALIENCY
# ===========================================================================

def compute_gradient_saliency(model, batch, device, target="pheno"):
    """
    梯度显著性分析：通过反向传播计算每个输入block对目标的重要性
    
    比注意力更精确地定位因果变异
    """
    model.eval()
    batch = {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in batch.items()}
    
    # 需要embedding层的梯度
    model.embedder.tok_emb.weight.requires_grad_(True)
    
    # 获取embedding输出
    emb = model.embedder(
        batch["hap_tokens"], batch["chr_ids"],
        batch["pos_ids"], batch["covariates"]
    )
    emb.retain_grad()
    
    # 前向传播（手动步骤）
    enc_out = model.transformer(emb)
    hidden = enc_out["hidden"]
    
    if target == "pheno":
        pheno_pred, _ = model.pheno_head(hidden)
        loss = pheno_pred.mean()
    elif target == "causal":
        snp_feat = hidden.mean(dim=1)
        expr_pred = model.expr_head(hidden)
        causal_out = model.causal(snp_feat, expr_pred)
        loss = causal_out["causal_pred"].mean()
    
    model.zero_grad()
    loss.backward()
    
    # 梯度 × 输入（Integrated Gradients简化版）
    if emb.grad is not None:
        saliency = (emb.grad * emb).abs().sum(dim=-1)  # (B, L)
        return saliency.detach().cpu().numpy().mean(axis=0)  # (L,)
    
    return None


# ===========================================================================
# 3. GENE IMPORTANCE ANALYSIS
# ===========================================================================

def compute_gene_scores_full(model, data_loader, device, gene_list) -> pd.DataFrame:
    """
    在完整数据集上汇总基因得分
    
    Returns:
        DataFrame with gene rankings and multiple importance scores
    """
    model.eval()
    all_gnn_scores = []
    all_causal_scores = []
    all_direct_effects = []
    all_indirect_effects = []
    all_alpha, all_beta = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            
            outputs = model(batch, stage="finetune")
            
            if "gene_scores" in outputs:
                all_gnn_scores.append(outputs["gene_scores"].cpu().numpy())
            if "causal_gene_scores" in outputs:
                all_causal_scores.append(outputs["causal_gene_scores"].cpu().numpy())
            if "causal_direct_effect" in outputs:
                all_direct_effects.append(outputs["causal_direct_effect"].cpu().numpy())
            if "causal_indirect_effect" in outputs:
                all_indirect_effects.append(outputs["causal_indirect_effect"].cpu().numpy())
            if "causal_alpha" in outputs:
                all_alpha.append(float(outputs["causal_alpha"]))
            if "causal_beta" in outputs:
                all_beta.append(float(outputs["causal_beta"]))
    
    n_genes = len(gene_list)
    
    # 汇总得分
    if all_gnn_scores:
        gnn_scores = np.array(all_gnn_scores).mean(axis=0)[:n_genes]
    else:
        gnn_scores = np.zeros(n_genes)
    
    if all_causal_scores:
        causal_scores = np.array(all_causal_scores).mean(axis=0)[:n_genes]
    else:
        causal_scores = np.zeros(n_genes)
    
    # 标准化得分到[0,1]
    def normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin > 1e-8:
            return (x - xmin) / (xmax - xmin)
        return np.zeros_like(x)
    
    gnn_norm = normalize(gnn_scores)
    causal_norm = normalize(causal_scores)
    combined = 0.5 * gnn_norm + 0.5 * causal_norm
    
    df = pd.DataFrame({
        "gene": gene_list[:n_genes],
        "gnn_score": gnn_scores,
        "gnn_score_norm": gnn_norm,
        "causal_score": causal_scores,
        "causal_score_norm": causal_norm,
        "combined_score": combined
    })
    
    # 全局路径权重
    if all_alpha:
        df["alpha_direct"] = np.mean(all_alpha)
        df["beta_indirect"] = np.mean(all_beta)
    
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    return df


# ===========================================================================
# 4. GENOMIC REGION MAPPING
# ===========================================================================

def map_attention_to_genome(block_attention: np.ndarray,
                             block_meta: list,
                             saliency: Optional[np.ndarray] = None,
                             top_k: int = 20) -> pd.DataFrame:
    """
    将block级别的注意力/显著性得分映射回基因组坐标
    
    Returns:
        DataFrame with top genomic regions
    """
    n_blocks = len(block_meta)
    n_scores = len(block_attention)
    n = min(n_blocks, n_scores)
    
    rows = []
    for i in range(n):
        meta = block_meta[i]
        row = {
            "chrom": meta["chrom"],
            "start": meta["start"],
            "end": meta["end"],
            "n_snps": meta["n_snps"],
            "attention_score": float(block_attention[i])
        }
        if saliency is not None and i < len(saliency):
            row["saliency_score"] = float(saliency[i])
            row["combined_importance"] = 0.5 * row["attention_score"] + 0.5 * row["saliency_score"]
        else:
            row["combined_importance"] = row["attention_score"]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 标准化
    for col in ["attention_score", "combined_importance"]:
        if col in df.columns:
            max_val = df[col].max()
            if max_val > 1e-8:
                df[f"{col}_norm"] = df[col] / max_val
    
    df = df.sort_values("combined_importance", ascending=False)
    df["region_rank"] = range(1, len(df) + 1)
    
    return df.head(top_k)


# ===========================================================================
# 5. CANDIDATE GENE TABLE (for wet lab)
# ===========================================================================

def generate_candidate_gene_table(gene_ranking: pd.DataFrame,
                                  region_ranking: pd.DataFrame,
                                  bim_df: pd.DataFrame,
                                  top_k: int = 30) -> pd.DataFrame:
    """
    生成湿实验候选基因表
    整合：GNN排名 + 基因组区域 + SNP位点信息
    
    输出列：
    - rank, gene_id, chromosome, approximate_position
    - gnn_score, causal_score, combined_score
    - top_region_overlap (是否在重要区域)
    - suggested_experiments (基于得分的建议实验)
    """
    top_genes = gene_ranking.head(top_k).copy()
    
    # 建议实验策略
    def suggest_experiment(row):
        if row["combined_score"] > 0.8:
            return "CRISPR-KO + qPCR验证（高优先级）"
        elif row["combined_score"] > 0.6:
            return "VIGS沉默 + 接种鉴定（中优先级）"
        elif row.get("beta_indirect", 0) > 0.5:
            return "转录因子结合分析 + Y2H互作（调控枢纽）"
        else:
            return "KASP标记开发 + 连锁分析"
    
    if "beta_indirect" in top_genes.columns:
        top_genes["suggested_experiment"] = top_genes.apply(suggest_experiment, axis=1)
    
    # 标记NBS-LRR相关（如果gene ID包含相关关键词）
    def infer_domain(gene_id):
        g = str(gene_id).upper()
        if any(x in g for x in ["NBS", "LRR", "TIR", "NLR"]):
            return "NBS-LRR (R gene)"
        elif any(x in g for x in ["WRKY"]):
            return "WRKY TF"
        elif any(x in g for x in ["RLK", "LRK"]):
            return "Receptor-like kinase"
        elif any(x in g for x in ["PR", "PATHOGENESIS"]):
            return "PR protein"
        else:
            return "Unknown"
    
    top_genes["predicted_domain"] = top_genes["gene"].apply(infer_domain)
    
    return top_genes


# ===========================================================================
# 6. VISUALIZATION
# ===========================================================================

def plot_training_history(history_path: str, save_path: str):
    """绘制训练曲线"""
    with open(history_path) as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("MHGT-GNN Training History", fontsize=14, fontweight="bold")
    
    metrics_to_plot = [
        ("mlm", "MLM Loss", "blue"),
        ("pheno", "Phenotype MSE", "green"),
        ("eqtl", "eQTL Loss", "orange"),
        ("causal", "Causal Loss", "red"),
        ("metric_r2", "R² (val)", "purple"),
        ("metric_pearson_r", "Pearson r (val)", "brown")
    ]
    
    for ax, (key, label, color) in zip(axes.flatten(), metrics_to_plot):
        train_vals = [e.get(key, np.nan) for e in history["train"]]
        val_vals = [e.get(key, np.nan) for e in history["val"]]
        epochs = list(range(1, len(train_vals) + 1))
        
        if not all(np.isnan(v) for v in train_vals):
            ax.plot(epochs, train_vals, color=color, alpha=0.7, label="train", linewidth=1.5)
        if not all(np.isnan(v) for v in val_vals):
            ax.plot(epochs, val_vals, color=color, linestyle="--", label="val", linewidth=1.5)
        
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history plot saved: {save_path}")


def plot_genome_heatmap(region_df: pd.DataFrame, save_path: str):
    """绘制全基因组注意力热图"""
    wheat_chrs = [f"{i}{g}" for i in range(1, 8) for g in ["A", "B", "D"]]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 为每条染色体分配颜色
    chr_colors = plt.cm.tab20(np.linspace(0, 1, len(wheat_chrs)))
    chr_color_map = dict(zip(wheat_chrs, chr_colors))
    
    # 绘制散点
    for _, row in region_df.iterrows():
        chrom = str(row["chrom"])
        x = row["start"] / 1e6  # Mb
        y = row.get("combined_importance_norm", row.get("attention_score", 0))
        color = chr_color_map.get(chrom, "gray")
        ax.scatter(x, y, color=color, alpha=0.6, s=30)
    
    # 标注Top 5
    top5 = region_df.head(5)
    for _, row in top5.iterrows():
        x = row["start"] / 1e6
        y = row.get("combined_importance_norm", row.get("attention_score", 0))
        ax.annotate(
            f"{row['chrom']}:{int(row['start']/1e6)}Mb",
            (x, y), textcoords="offset points", xytext=(5, 5),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.5)
        )
    
    ax.set_xlabel("Physical Position (Mb)", fontsize=11)
    ax.set_ylabel("Attention Importance", fontsize=11)
    ax.set_title("Genome-wide Attention Map - Wheat Stripe Rust Resistance",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Genome heatmap saved: {save_path}")


def plot_gene_ranking(gene_df: pd.DataFrame, top_k: int = 30, save_path: str = ""):
    """绘制候选基因排名图"""
    top = gene_df.head(top_k).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_k * 0.3)))
    
    # 左：GNN分数条形图
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))[::-1]
    bars = ax.barh(range(len(top)), top["gnn_score_norm"] if "gnn_score_norm" in top.columns
                   else top["gnn_score"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("GNN Score (normalized)", fontsize=10)
    ax.set_title("GNN Gene Prioritization", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 右：因果vs直接效应散点
    ax2 = axes[1]
    if "causal_score_norm" in top.columns and "gnn_score_norm" in top.columns:
        scatter = ax2.scatter(
            top["gnn_score_norm"],
            top["causal_score_norm"],
            c=top["combined_score"] if "combined_score" in top.columns else "blue",
            cmap="viridis", s=60, alpha=0.8
        )
        plt.colorbar(scatter, ax=ax2, label="Combined Score")
        
        # 标注Top 10
        for _, row in top.head(10).iterrows():
            ax2.annotate(row["gene"],
                        (row["gnn_score_norm"], row["causal_score_norm"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=6)
        
        ax2.set_xlabel("GNN Score", fontsize=10)
        ax2.set_ylabel("Causal Score", fontsize=10)
        ax2.set_title("GNN vs Causal Score", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Gene ranking plot saved: {save_path}")
    return fig


def plot_causal_decomposition(direct_vals: np.ndarray,
                               indirect_vals: np.ndarray,
                               pheno_vals: np.ndarray,
                               save_path: str):
    """可视化直接效应vs间接效应（因果分解）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 直接效应 vs 表型
    ax = axes[0]
    ax.scatter(pheno_vals, direct_vals, alpha=0.5, s=20, color="#2E86AB")
    m = np.polyfit(pheno_vals, direct_vals, 1)
    x_line = np.linspace(pheno_vals.min(), pheno_vals.max(), 100)
    ax.plot(x_line, np.polyval(m, x_line), "r--", linewidth=2)
    r = np.corrcoef(pheno_vals, direct_vals)[0, 1]
    ax.set_title(f"Direct Effect\n(r={r:.3f})", fontsize=11)
    ax.set_xlabel("Phenotype (BLUP)")
    ax.set_ylabel("Direct Genetic Effect")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. 间接效应 vs 表型
    ax = axes[1]
    ax.scatter(pheno_vals, indirect_vals, alpha=0.5, s=20, color="#A23B72")
    m = np.polyfit(pheno_vals, indirect_vals, 1)
    ax.plot(x_line, np.polyval(m, x_line), "r--", linewidth=2)
    r = np.corrcoef(pheno_vals, indirect_vals)[0, 1]
    ax.set_title(f"Indirect Effect (via expression)\n(r={r:.3f})", fontsize=11)
    ax.set_xlabel("Phenotype (BLUP)")
    ax.set_ylabel("Indirect Effect")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. 效应分解饼图（以均值为准）
    ax = axes[2]
    direct_mean = abs(direct_vals).mean()
    indirect_mean = abs(indirect_vals).mean()
    total = direct_mean + indirect_mean
    if total > 0:
        sizes = [direct_mean / total, indirect_mean / total]
        labels = [f"Direct\n({direct_mean/total*100:.1f}%)", f"Indirect\n({indirect_mean/total*100:.1f}%)"]
        colors = ["#2E86AB", "#A23B72"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
    ax.set_title("Causal Effect Decomposition", fontsize=11)
    
    plt.suptitle("Causal Inference Analysis - Stripe Rust Resistance",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Causal decomposition plot saved: {save_path}")


# ===========================================================================
# 7. MAIN INTERPRETATION PIPELINE
# ===========================================================================

def run_interpretation(model, data_loader, processed_data, device, output_dir):
    """完整解释流程"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gene_list = processed_data.get("gene_list", [])
    block_meta = processed_data.get("block_meta", [])
    
    logger.info("Step 1: Extracting attention maps...")
    attn_maps = extract_attention_maps(model, data_loader, device, n_batches=10)
    
    logger.info("Step 2: Computing gradient saliency...")
    saliency = None
    try:
        batch = next(iter(data_loader))
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        saliency = compute_gradient_saliency(model, batch, device, target="pheno")
    except Exception as e:
        logger.warning(f"Gradient saliency failed: {e}")
    
    logger.info("Step 3: Computing gene rankings...")
    if gene_list:
        gene_df = compute_gene_scores_full(model, data_loader, device, gene_list)
        gene_df.to_csv(output_dir / "gene_ranking.csv", index=False)
        logger.info(f"Top 10 candidate genes:\n{gene_df.head(10).to_string()}")
    else:
        gene_df = pd.DataFrame()
        logger.warning("No gene list available for ranking")
    
    logger.info("Step 4: Mapping to genome coordinates...")
    if "pheno_attention" in attn_maps and block_meta:
        pheno_attn = attn_maps["pheno_attention"]
        region_df = map_attention_to_genome(pheno_attn, block_meta, saliency, top_k=50)
        region_df.to_csv(output_dir / "top_regions.csv", index=False)
    else:
        region_df = pd.DataFrame()
    
    logger.info("Step 5: Generating candidate gene table for wet lab...")
    if not gene_df.empty and not region_df.empty:
        bim_df = pd.DataFrame(processed_data.get("bim", {}))
        candidate_df = generate_candidate_gene_table(gene_df, region_df, bim_df)
        candidate_df.to_csv(output_dir / "candidate_genes_wetlab.csv", index=False)
        logger.info(f"\n{'='*60}")
        logger.info("TOP CANDIDATE GENES FOR EXPERIMENTAL VALIDATION:")
        logger.info(f"{'='*60}")
        display_cols = ["rank", "gene", "combined_score", "predicted_domain",
                       "suggested_experiment"] if "suggested_experiment" in candidate_df.columns \
                       else ["rank", "gene", "combined_score"]
        logger.info(candidate_df[display_cols].head(20).to_string(index=False))
    
    logger.info("Step 6: Generating visualizations...")
    if "pheno_attention" in attn_maps and not region_df.empty:
        plot_genome_heatmap(region_df, str(output_dir / "genome_attention.png"))
    
    if not gene_df.empty:
        plot_gene_ranking(gene_df, top_k=30, save_path=str(output_dir / "gene_ranking.png"))
    
    # 收集因果效应数据用于可视化
    try:
        all_direct, all_indirect, all_pheno = collect_causal_effects(model, data_loader, device)
        if len(all_direct) > 0:
            plot_causal_decomposition(
                np.array(all_direct), np.array(all_indirect), np.array(all_pheno),
                str(output_dir / "causal_decomposition.png")
            )
    except Exception as e:
        logger.warning(f"Causal decomposition plot failed: {e}")
    
    logger.info(f"\n✅ Interpretation complete. Results in: {output_dir}")
    return {
        "gene_ranking": gene_df,
        "top_regions": region_df,
        "attention_maps": attn_maps
    }


@torch.no_grad()
def collect_causal_effects(model, data_loader, device):
    """收集所有样本的直接/间接效应用于可视化"""
    model.eval()
    all_direct, all_indirect, all_pheno = [], [], []
    
    for batch in data_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        outputs = model(batch, stage="finetune")
        
        if "causal_direct_effect" in outputs:
            all_direct.extend(outputs["causal_direct_effect"].cpu().numpy().flatten())
        if "causal_indirect_effect" in outputs:
            all_indirect.extend(outputs["causal_indirect_effect"].cpu().numpy().flatten())
        if "pheno" in batch:
            all_pheno.extend(batch["pheno"].cpu().numpy().flatten())
    
    return all_direct, all_indirect, all_pheno


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHGT-GNN Interpretation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processed_data", type=str,
                        default="./data/processed/processed_data.pkl")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/interpretation")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    # 动态导入（避免相对路径问题）
    import importlib.util, sys
    
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    
    base = Path(__file__).parent
    preprocess = load_module("preprocess_01", base / "01_preprocess.py")
    model_mod = load_module("model_02", base / "02_model.py")
    
    cfg = preprocess.load_config(args.config)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    dataset, processed_data = preprocess.load_processed_dataset(args.processed_data)
    _, val_loader, test_loader, _ = preprocess.create_dataloaders(
        dataset, cfg, batch_size=args.batch_size
    )
    
    # 加载模型
    model = model_mod.build_model(cfg, processed_data)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)
    logger.info(f"Model loaded from {args.checkpoint}")
    
    # 运行解释
    run_interpretation(model, test_loader, processed_data, device, args.output_dir)
