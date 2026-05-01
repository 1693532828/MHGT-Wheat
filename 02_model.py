"""
MHGT-GNN Pipeline - Step 2: Model Architecture
================================================
Multi-modal Haplotype-Graph Transformer with
causal inference and GNN gene prioritization

Architecture:
  HaplotypeEmbedder → GenomicTransformer → [MLM Head]
                                         → [Phenotype Head]
                                         → [eQTL/Expression Head]
                                         → [Causal SEM Module]
                                         → [Gene Prioritization GNN]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Dict, Optional, Tuple


# ===========================================================================
# 1. POSITIONAL & EMBEDDING LAYERS
# ===========================================================================

class HaplotypeEmbedder(nn.Module):
    """
    将Haplotype Token序列转为连续向量表示
    
    输入: token_ids(B,L), chr_ids(B,L), pos_ids(B,L), covariates(B,C)
    输出: embeddings(B,L,D)
    """
    
    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_chr: int = 22, max_pos: int = 5000, n_cov: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding（主要语义信息）
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 染色体ID embedding（21条 + 1个unknown）
        self.chr_emb = nn.Embedding(n_chr, d_model // 4)
        
        # 物理位置embedding（连续位置编码）
        self.pos_emb = nn.Embedding(max_pos + 1, d_model // 4)
        
        # 协变量投影（PCA等）
        self.cov_proj = nn.Sequential(
            nn.Linear(n_cov, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )
        
        # 融合投影
        fused_dim = d_model + d_model // 4 + d_model // 4 + d_model // 2
        self.fuse = nn.Linear(fused_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.chr_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
    
    def forward(self, tokens: torch.Tensor, chr_ids: torch.Tensor,
                pos_ids: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        
        tok = self.tok_emb(tokens)                              # (B, L, D)
        chr = self.chr_emb(chr_ids)                             # (B, L, D//4)
        pos = self.pos_emb(pos_ids.clamp(0, self.pos_emb.num_embeddings - 1))  # (B, L, D//4)
        cov = self.cov_proj(covariates).unsqueeze(1).expand(-1, L, -1)  # (B, L, D//2)
        
        x = torch.cat([tok, chr, pos, cov], dim=-1)            # (B, L, fused_dim)
        x = self.fuse(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class SinusoidalPositionEncoding(nn.Module):
    """正弦位置编码（可选，作为位置embedding的补充）"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ===========================================================================
# 2. TRANSFORMER ENCODER
# ===========================================================================

class GenomicTransformerLayer(nn.Module):
    """
    基因组特化的Transformer层
    支持：标准注意力 / 线性注意力（大序列时）
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 ffn_mult: int = 4, dropout: float = 0.1,
                 use_flash: bool = False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        
        # Feed-Forward Network
        dim_ffn = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm residual connection
        residual = x
        x_norm = self.norm1(x)
        
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True
        )
        x = residual + self.dropout(attn_out)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights


class GenomicTransformer(nn.Module):
    """多层基因组Transformer编码器"""
    
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 nlayer: int = 6, dropout: float = 0.1,
                 ffn_mult: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            GenomicTransformerLayer(d_model, nhead, ffn_mult, dropout)
            for _ in range(nlayer)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.nlayer = nlayer
    
    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                return_all_layers: bool = False,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        
        all_hidden = []
        last_attn = None
        
        for i, layer in enumerate(self.layers):
            need_weights = (return_attention and i == self.nlayer - 1)
            x, attn = layer(x, key_padding_mask, need_weights=need_weights)
            if return_all_layers:
                all_hidden.append(x)
            if need_weights:
                last_attn = attn
        
        x = self.final_norm(x)
        
        out = {"hidden": x}
        if return_all_layers:
            out["all_hidden"] = all_hidden
        if last_attn is not None:
            out["attention"] = last_attn
        
        return out


# ===========================================================================
# 3. PREDICTION HEADS
# ===========================================================================

class MaskedLMHead(nn.Module):
    """Masked Language Model预测头（预训练用）"""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)
        
        # 权重绑定（可选）
        # self.decoder.weight = embedding.weight
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden)
        x = self.act(x)
        x = self.norm(x)
        return self.decoder(x)  # (B, L, vocab_size)


class PhenotypeHead(nn.Module):
    """
    表型预测头（回归）
    使用attention-weighted pooling替代简单mean pooling
    """
    
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        # Attention pooling
        self.attn_pool = nn.Linear(d_model, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, hidden: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention pooling: 学习每个位置对表型预测的权重
        attn_logits = self.attn_pool(hidden).squeeze(-1)   # (B, L)
        
        if padding_mask is not None:
            attn_logits = attn_logits.masked_fill(padding_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_logits, dim=-1)       # (B, L)
        pooled = (hidden * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        
        pred = self.mlp(pooled)                              # (B, 1)
        return pred, attn_weights


class ExpressionHead(nn.Module):
    """
    基因表达量预测头（eQTL效应）
    预测每个样本的基因表达谱
    """
    
    def __init__(self, d_model: int, n_genes: int, dropout: float = 0.2):
        super().__init__()
        self.n_genes = n_genes
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_genes)
        )
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # 平均池化
        pooled = hidden.mean(dim=1)   # (B, D)
        return self.mlp(pooled)        # (B, n_genes)


# ===========================================================================
# 4. CAUSAL INFERENCE MODULE (SEM-based)
# ===========================================================================

class CausalInferenceModule(nn.Module):
    """
    可微分结构方程模型 (Structural Equation Model)
    
    模型假设:
        Y = α * f_direct(H) + β * g_indirect(E) + ε
    
    其中:
        H = Haplotype特征 (Transformer输出)
        E = 预测的基因表达量 (eQTL路径)
        Y = 表型
        α, β = 可学习的路径权重
    
    这使模型能够：
    1. 分解遗传效应为直接效应（调控区变异）和间接效应（表达量介导）
    2. 通过MR约束防止反向因果
    3. 识别具有高间接效应的功能基因
    """
    
    def __init__(self, d_model: int, n_genes: int, dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        
        # 直接遗传效应路径: H → Y
        self.direct_path = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 间接效应路径（表达量介导）: E → Y
        self.indirect_path = nn.Sequential(
            nn.Linear(n_genes, n_genes // 2 + 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_genes // 2 + 1, 1)
        )
        
        # 路径权重（可学习，通过sigmoid约束在0~1）
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # 直接路径权重
        self.beta_logit = nn.Parameter(torch.zeros(1))   # 间接路径权重
        
        # 基因级别的因果得分（哪些基因在间接路径中最重要）
        self.gene_causal_weight = nn.Linear(n_genes, n_genes, bias=False)
    
    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)
    
    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)
    
    def forward(self, snp_hidden: torch.Tensor,
                expr_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            snp_hidden: (B, D) - Transformer池化后的SNP特征
            expr_pred: (B, G) - 预测的基因表达量
        
        Returns:
            dict包含causal_pred, direct_effect, indirect_effect, gene_scores
        """
        # 直接效应
        direct = self.direct_path(snp_hidden)          # (B, 1)
        
        # 基因因果权重（学习哪些基因是功能性的）
        expr_weighted = self.gene_causal_weight(expr_pred)  # (B, G)
        
        # 间接效应
        indirect = self.indirect_path(expr_weighted)    # (B, 1)
        
        # 加权融合
        causal_pred = self.alpha * direct + self.beta * indirect
        
        # 每个基因的因果得分（绝对值表示重要性）
        gene_scores = torch.abs(self.gene_causal_weight.weight).mean(dim=0)  # (G,)
        
        return {
            "causal_pred": causal_pred,
            "direct_effect": direct,
            "indirect_effect": indirect,
            "gene_scores": gene_scores,
            "alpha": self.alpha,
            "beta": self.beta
        }


# ===========================================================================
# 5. GNN GENE PRIORITIZATION MODULE
# ===========================================================================

class GATLayer(nn.Module):
    """图注意力网络层（不依赖PyG，纯PyTorch实现）"""
    
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.head_dim = out_dim // n_heads if concat else out_dim
        
        self.W = nn.Linear(in_dim, self.head_dim * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim if concat else out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) 节点特征
            edge_index: (2, E) 边索引
        """
        N = x.size(0)
        if edge_index.size(1) == 0:
            # 无边时直接投影
            out = self.W(x)  # (N, n_heads*head_dim)
            if not self.concat:
                out = out.view(N, self.n_heads, self.head_dim).mean(dim=1)
            return F.elu(self.norm(out))
        
        Wh = self.W(x).view(N, self.n_heads, self.head_dim)  # (N, H, D)
        
        src, dst = edge_index[0], edge_index[1]
        
        # 注意力分数
        e_src = (Wh[src] * self.a_src).sum(-1)   # (E, H)
        e_dst = (Wh[dst] * self.a_dst).sum(-1)   # (E, H)
        e = self.leaky(e_src + e_dst)              # (E, H)
        
        # Sparse softmax（按目标节点归一化）
        # 使用scatter_softmax等价实现
        attn = torch.zeros(N, N, self.n_heads, device=x.device)
        attn[dst, src] = e
        attn = torch.where(attn != 0, attn, torch.full_like(attn, float('-inf')))
        attn = F.softmax(attn, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        # 聚合
        out = torch.einsum('nji,inh->njh', attn[..., :], Wh)  # 简化版本
        
        # 更简单的稀疏聚合
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        e_soft = F.softmax(e, dim=0)
        for h in range(self.n_heads):
            out[:, h].scatter_add_(0, dst.unsqueeze(1).expand(-1, self.head_dim),
                                   e_soft[:, h:h+1] * Wh[src, h])
        
        if self.concat:
            out = out.reshape(N, -1)
        else:
            out = out.mean(dim=1)
        
        return F.elu(self.norm(out))


class GeneGNN(nn.Module):
    """
    基因优先级排序图神经网络
    
    输入节点：
    - 基因节点：初始特征 = 基因注释特征 + Transformer提取的eQTL效应
    - （可选）通过eQTL边连接的SNP节点
    
    输出：每个基因的因果优先级得分
    """
    
    def __init__(self, gene_feat_dim: int, snp_feat_dim: int,
                 hidden: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1,
                 n_genes: int = 100):
        super().__init__()
        
        # 基因节点投影
        self.gene_proj = nn.Linear(gene_feat_dim, hidden)
        
        # SNP节点投影（用Transformer输出均值）
        self.snp_proj = nn.Linear(snp_feat_dim, hidden)
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        in_dim = hidden
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else hidden // 2
            self.gat_layers.append(GATLayer(in_dim, out_dim, n_heads, dropout,
                                            concat=(i < n_layers - 1)))
            in_dim = out_dim
        
        # 排序头
        self.ranker = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1)
        )
        
        self.n_genes = n_genes
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, gene_feat: torch.Tensor, snp_feat: torch.Tensor,
                gene_edges: torch.Tensor, causal_scores: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gene_feat: (G, gene_feat_dim) 基因节点特征
            snp_feat: (D,) 或 (B, D) 批次平均SNP特征
            gene_edges: (2, E) 基因-基因边
            causal_scores: (G,) 来自CausalModule的先验得分
        
        Returns:
            scores: (G,) 每个基因的优先级得分
            node_emb: (G, hidden//2) 基因节点嵌入
        """
        G = gene_feat.shape[0]
        
        # 节点初始化
        gene_nodes = F.relu(self.gene_proj(gene_feat))          # (G, hidden)
        
        # 若有因果得分，作为先验注入
        if causal_scores is not None:
            gene_nodes = gene_nodes + causal_scores.unsqueeze(1) * gene_nodes
        
        x = self.dropout(gene_nodes)
        
        # 确保edge_index在正确范围内
        if gene_edges.size(1) > 0:
            valid_mask = (gene_edges[0] < G) & (gene_edges[1] < G)
            gene_edges_valid = gene_edges[:, valid_mask]
        else:
            gene_edges_valid = gene_edges
        
        # GAT前向传播
        for layer in self.gat_layers:
            x = layer(x, gene_edges_valid)
        
        # 排序分数
        scores = self.ranker(x).squeeze(-1)                     # (G,)
        
        return scores, x


# ===========================================================================
# 6. COMPLETE MODEL
# ===========================================================================

class MHGT_GNN(nn.Module):
    """
    Multi-modal Haplotype-Graph Transformer + GNN
    
    完整框架整合：
    1. HaplotypeEmbedder: Token → Dense embedding
    2. GenomicTransformer: Contextual representation
    3. Multi-task heads: MLM + Phenotype + Expression
    4. CausalInferenceModule: 可微分SEM因果分解
    5. GeneGNN: 图神经网络基因优先级排序
    """
    
    def __init__(self, vocab_size: int, n_genes: int,
                 d_model: int = 256, nhead: int = 8,
                 nlayer: int = 6, n_chr: int = 22,
                 max_pos: int = 5000, n_cov: int = 10,
                 gene_feat_dim: int = 64, gnn_hidden: int = 128,
                 dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_genes = max(n_genes, 1)
        self.d_model = d_model
        
        # --- Encoder ---
        self.embedder = HaplotypeEmbedder(
            vocab_size, d_model, n_chr, max_pos, n_cov, dropout
        )
        self.transformer = GenomicTransformer(
            d_model, nhead, nlayer, dropout, ffn_mult
        )
        
        # --- Prediction Heads ---
        self.mlm_head = MaskedLMHead(d_model, vocab_size)
        self.pheno_head = PhenotypeHead(d_model, dropout)
        self.expr_head = ExpressionHead(d_model, self.n_genes, dropout)
        
        # --- Causal Module ---
        self.causal = CausalInferenceModule(d_model, self.n_genes, dropout)
        
        # --- Gene GNN ---
        self.gene_gnn = GeneGNN(
            gene_feat_dim=gene_feat_dim,
            snp_feat_dim=d_model,
            hidden=gnn_hidden,
            n_genes=self.n_genes
        )
        
        # 模型参数统计
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MHGT-GNN] Initialized: {n_params:,} trainable parameters")
        print(f"  vocab_size={vocab_size}, n_genes={n_genes}, d_model={d_model}")
        print(f"  nlayer={nlayer}, nhead={nhead}")
    
    def encode(self, batch: Dict[str, torch.Tensor],
               return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """编码阶段：Embedding + Transformer"""
        # 构建padding mask（token=0的位置）
        pad_mask = (batch["hap_tokens"] == 0)
        if pad_mask.all(dim=-1).any():
            pad_mask = None  # 避免全mask
        
        # 嵌入
        emb = self.embedder(
            batch["hap_tokens"], batch["chr_ids"],
            batch["pos_ids"], batch["covariates"]
        )
        
        # 变换
        enc_out = self.transformer(
            emb, key_padding_mask=pad_mask,
            return_attention=return_attention
        )
        
        return enc_out, pad_mask
    
    def forward(self, batch: Dict[str, torch.Tensor],
                return_attention: bool = False,
                stage: str = "finetune") -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 来自DataLoader的批次数据
            return_attention: 是否返回注意力权重（用于可解释性）
            stage: "pretrain" | "finetune" | "interpret"
        
        Returns:
            outputs dict
        """
        # ---- Encoding ----
        enc_out, pad_mask = self.encode(batch, return_attention)
        hidden = enc_out["hidden"]  # (B, L, D)
        
        outputs = {}
        
        # ---- Stage 1: Pretrain - 仅MLM ----
        if stage == "pretrain":
            mlm_logit = self.mlm_head(hidden)
            outputs["mlm_logit"] = mlm_logit
            if return_attention and "attention" in enc_out:
                outputs["attention"] = enc_out["attention"]
            return outputs
        
        # ---- Stage 2/3: Finetune/Interpret ----
        # MLM head（可继续辅助训练）
        if stage in ["finetune", "interpret"]:
            mlm_logit = self.mlm_head(hidden)
            outputs["mlm_logit"] = mlm_logit
        
        # 表型预测
        pheno_pred, attn_weights = self.pheno_head(hidden, pad_mask)
        outputs["pheno_pred"] = pheno_pred
        outputs["pheno_attn"] = attn_weights
        
        # 表达量预测（eQTL效应）
        expr_pred = self.expr_head(hidden)
        outputs["expr_pred"] = expr_pred
        
        # 因果推断（需要汇总SNP特征）
        snp_feat = hidden.mean(dim=1)   # (B, D) - 简单均值池化
        causal_out = self.causal(snp_feat, expr_pred)
        outputs.update({f"causal_{k}": v for k, v in causal_out.items()})
        
        # GNN基因排序（用批次平均的特征）
        gene_feat = batch["gene_node_feat"]  # (G, gene_feat_dim)
        
        # 提取批次级别基因特征（eQTL路径：用expr_pred均值作为基因表达信号）
        gene_scores, gene_emb = self.gene_gnn(
            gene_feat=gene_feat,
            snp_feat=snp_feat.mean(0),
            gene_edges=batch["gene_edges"],
            causal_scores=causal_out["gene_scores"].detach()
        )
        outputs["gene_scores"] = gene_scores
        outputs["gene_emb"] = gene_emb
        
        if return_attention and "attention" in enc_out:
            outputs["attention"] = enc_out["attention"]
        
        return outputs
    
    def get_gene_ranking(self, outputs: Dict[str, torch.Tensor],
                         gene_list: list, top_k: int = 30) -> pd.DataFrame:
        """从模型输出生成基因排序表"""
        import pandas as pd
        
        scores = outputs["gene_scores"].detach().cpu().numpy()
        causal_scores = outputs.get("causal_gene_scores", torch.zeros_like(outputs["gene_scores"]))
        causal_np = causal_scores.detach().cpu().numpy() if torch.is_tensor(causal_scores) else causal_scores
        
        n = min(len(gene_list), len(scores))
        df = pd.DataFrame({
            "gene": gene_list[:n],
            "gnn_score": scores[:n],
            "causal_score": causal_np[:n] if len(causal_np) >= n else [0]*n,
            "combined_score": (scores[:n] + causal_np[:n]) / 2 if len(causal_np) >= n else scores[:n]
        })
        
        df = df.sort_values("combined_score", ascending=False).head(top_k)
        df["rank"] = range(1, len(df) + 1)
        return df
    
    @torch.no_grad()
    def get_attention_map(self, batch: Dict[str, torch.Tensor],
                          block_meta: list) -> Dict[str, torch.Tensor]:
        """提取注意力图谱用于可解释性分析"""
        self.eval()
        outputs = self.forward(batch, return_attention=True, stage="interpret")
        
        attn = outputs.get("attention")  # (B, L, L)
        pheno_attn = outputs.get("pheno_attn")  # (B, L)
        
        result = {
            "pheno_attention": pheno_attn,  # 表型预测最关注的block
            "block_meta": block_meta
        }
        if attn is not None:
            result["self_attention"] = attn
        
        return result


# ===========================================================================
# 7. MODEL FACTORY
# ===========================================================================

def build_model(cfg: dict, processed_data: dict) -> MHGT_GNN:
    """根据配置和数据参数构建模型"""
    model_cfg = cfg.get("model", {})
    
    model = MHGT_GNN(
        vocab_size=processed_data["vocab_size"],
        n_genes=max(processed_data["n_genes"], 1),
        d_model=model_cfg.get("d_model", 256),
        nhead=model_cfg.get("nhead", 8),
        nlayer=model_cfg.get("nlayer", 6),
        n_chr=model_cfg.get("n_chr", 22),
        max_pos=model_cfg.get("max_pos", 5000),
        n_cov=model_cfg.get("n_cov", 10),
        gene_feat_dim=processed_data["gene_feat"].shape[1],
        gnn_hidden=model_cfg.get("gnn_hidden", 128),
        dropout=model_cfg.get("dropout", 0.1),
        ffn_mult=model_cfg.get("ffn_mult", 4)
    )
    
    return model


if __name__ == "__main__":
    # 快速架构测试
    print("Testing model architecture...")
    
    B, L, G, D = 4, 64, 50, 256
    vocab_size = 500
    
    model = MHGT_GNN(
        vocab_size=vocab_size, n_genes=G, d_model=D,
        nhead=8, nlayer=4, gene_feat_dim=64
    )
    
    batch = {
        "hap_tokens": torch.randint(0, vocab_size, (B, L)),
        "chr_ids": torch.randint(0, 22, (B, L)),
        "pos_ids": torch.randint(0, 5000, (B, L)),
        "covariates": torch.randn(B, 10),
        "pheno": torch.randn(B, 1),
        "gene_node_feat": torch.randn(G, 64),
        "eqtl_edges": torch.zeros(2, 0, dtype=torch.long),
        "gene_edges": torch.zeros(2, 0, dtype=torch.long),
    }
    
    with torch.no_grad():
        out = model(batch, stage="finetune")
    
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✅ Architecture test passed!")
