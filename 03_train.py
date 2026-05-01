"""
MHGT-GNN Pipeline - Step 3: Training Engine
============================================
Multi-task loss functions + two-stage training loop
(Stage 1: MLM pretrain / Stage 2: Multi-task finetune)

Usage:
    python 03_train.py --config config.yaml --stage pretrain
    python 03_train.py --config config.yaml --stage finetune --checkpoint ./checkpoints/pretrain_best.pt
"""

import os
import json
import time
import logging
import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

# 内部模块
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_01 import load_processed_dataset, create_dataloaders, load_config
from model_02 import build_model, MHGT_GNN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# 1. LOSS FUNCTIONS
# ===========================================================================

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    包含：
    1. MLM Loss       - 掩码语言模型（预训练/辅助）
    2. Phenotype Loss - 表型回归MSE
    3. eQTL Loss      - 表达量预测MSE（仅有表达数据的样本）
    4. Causal Loss    - 因果模型约束
    5. MR Reg         - Mendelian Randomization符号一致性正则
    6. DAG Reg        - 有向无环图约束（防止循环因果）
    
    损失权重通过不确定性加权自动学习（Kendall et al. 2018）
    """
    
    def __init__(self, lambda_mlm: float = 1.0, lambda_pheno: float = 1.0,
                 lambda_eqtl: float = 0.5, lambda_causal: float = 0.3,
                 lambda_mr: float = 0.1, use_uncertainty: bool = True):
        super().__init__()
        
        self.lambda_mlm = lambda_mlm
        self.lambda_pheno = lambda_pheno
        self.lambda_eqtl = lambda_eqtl
        self.lambda_causal = lambda_causal
        self.lambda_mr = lambda_mr
        
        # 不确定性加权参数（log sigma^2）
        if use_uncertainty:
            self.log_sigma_mlm = nn.Parameter(torch.zeros(1))
            self.log_sigma_pheno = nn.Parameter(torch.zeros(1))
            self.log_sigma_eqtl = nn.Parameter(torch.zeros(1))
            self.log_sigma_causal = nn.Parameter(torch.zeros(1))
        else:
            self.log_sigma_mlm = None
        
        self.use_uncertainty = use_uncertainty
    
    def uncertainty_weight(self, loss: torch.Tensor,
                           log_sigma: nn.Parameter) -> torch.Tensor:
        """L_weighted = L / (2 * sigma^2) + log(sigma)"""
        precision = torch.exp(-log_sigma)
        return precision * loss + log_sigma * 0.5
    
    def mlm_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """只在被mask的位置计算交叉熵"""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        flat_mask = mask.view(-1)
        return F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])
    
    def eqtl_loss(self, expr_pred: torch.Tensor, expr_true: torch.Tensor,
                  has_expr: Optional[torch.Tensor]) -> torch.Tensor:
        """仅对有表达数据的样本计算损失"""
        if expr_true is None:
            return torch.tensor(0.0, device=expr_pred.device)
        if has_expr is not None:
            if has_expr.sum() == 0:
                return torch.tensor(0.0, device=expr_pred.device)
            expr_pred = expr_pred[has_expr]
            expr_true = expr_true[has_expr]
        return F.mse_loss(expr_pred, expr_true)
    
    def causal_loss(self, causal_pred: torch.Tensor,
                    pheno_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(causal_pred, pheno_true)
    
    def mr_regularization(self, indirect_effect: torch.Tensor,
                          pheno: torch.Tensor) -> torch.Tensor:
        """
        MR约束：间接效应方向应与表型方向一致
        惩罚方向相反的路径（防止反向因果混淆）
        """
        # indirect_effect: (B, 1), pheno: (B, 1)
        sign_conflict = F.relu(-indirect_effect * pheno)
        return sign_conflict.mean()
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor],
                stage: str = "finetune") -> Tuple[torch.Tensor, Dict[str, float]]:
        
        device = next(iter(outputs.values())).device
        losses = {}
        
        # 1. MLM Loss
        if "mlm_logit" in outputs and "mask" in batch:
            l_mlm = self.mlm_loss(outputs["mlm_logit"], batch["labels"], batch["mask"])
        else:
            l_mlm = torch.tensor(0.0, device=device)
        losses["mlm"] = l_mlm.item()
        
        if stage == "pretrain":
            total = self.lambda_mlm * l_mlm
            return total, losses
        
        # 2. Phenotype Loss
        if "pheno" in batch:
            l_pheno = F.mse_loss(outputs["pheno_pred"], batch["pheno"])
            losses["pheno"] = l_pheno.item()
        else:
            l_pheno = torch.tensor(0.0, device=device)
            losses["pheno"] = 0.0
        
        # 3. eQTL Loss
        expr_true = batch.get("expr", None)
        has_expr = batch.get("has_expr", None)
        l_eqtl = self.eqtl_loss(outputs["expr_pred"], expr_true, has_expr)
        losses["eqtl"] = l_eqtl.item()
        
        # 4. Causal Loss
        if "causal_causal_pred" in outputs and "pheno" in batch:
            l_causal = self.causal_loss(outputs["causal_causal_pred"], batch["pheno"])
            losses["causal"] = l_causal.item()
        else:
            l_causal = torch.tensor(0.0, device=device)
            losses["causal"] = 0.0
        
        # 5. MR Regularization
        if "causal_indirect_effect" in outputs and "pheno" in batch:
            l_mr = self.mr_regularization(outputs["causal_indirect_effect"], batch["pheno"])
            losses["mr_reg"] = l_mr.item()
        else:
            l_mr = torch.tensor(0.0, device=device)
            losses["mr_reg"] = 0.0
        
        # 组合损失
        if self.use_uncertainty and self.log_sigma_mlm is not None:
            total = (
                self.uncertainty_weight(l_mlm, self.log_sigma_mlm) +
                self.uncertainty_weight(l_pheno, self.log_sigma_pheno) +
                self.uncertainty_weight(l_eqtl + l_causal, self.log_sigma_eqtl) +
                self.uncertainty_weight(l_causal + self.lambda_mr * l_mr, self.log_sigma_causal)
            )
        else:
            total = (
                self.lambda_mlm * l_mlm +
                self.lambda_pheno * l_pheno +
                self.lambda_eqtl * l_eqtl +
                self.lambda_causal * (l_causal + self.lambda_mr * l_mr)
            )
        
        losses["total"] = total.item()
        return total, losses


# ===========================================================================
# 2. METRICS
# ===========================================================================

def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    """计算回归指标：MSE, MAE, R², Pearson r"""
    pred_np = pred.detach().cpu().numpy().flatten()
    true_np = true.detach().cpu().numpy().flatten()
    
    mse = float(np.mean((pred_np - true_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - true_np)))
    
    # R²
    ss_res = np.sum((true_np - pred_np) ** 2)
    ss_tot = np.sum((true_np - true_np.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    # Pearson r
    if true_np.std() > 0 and pred_np.std() > 0:
        pearson_r = float(np.corrcoef(pred_np, true_np)[0, 1])
    else:
        pearson_r = 0.0
    
    return {"mse": mse, "mae": mae, "r2": r2, "pearson_r": pearson_r}


# ===========================================================================
# 3. TRAINER
# ===========================================================================

class Trainer:
    """
    两阶段训练器
    
    Stage 1 (Pretrain): 仅MLM，利用所有样本（含无表型）
    Stage 2 (Finetune): 多任务联合优化
    """
    
    def __init__(self, model: MHGT_GNN, cfg: dict,
                 device: torch.device, checkpoint_dir: str):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        train_cfg = cfg.get("training", {})
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.log_interval = train_cfg.get("log_interval", 20)
        
        self.loss_fn = MultiTaskLoss(
            lambda_mlm=train_cfg.get("lambda_mlm", 1.0),
            lambda_pheno=train_cfg.get("lambda_pheno", 1.0),
            lambda_eqtl=train_cfg.get("lambda_eqtl", 0.5),
            lambda_causal=train_cfg.get("lambda_causal", 0.3),
            lambda_mr=train_cfg.get("lambda_mr", 0.1),
            use_uncertainty=train_cfg.get("use_uncertainty", True)
        ).to(device)
        
        self.history = {"train": [], "val": []}
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.patience = train_cfg.get("patience", 15)
    
    def build_optimizer(self, stage: str) -> Tuple[torch.optim.Optimizer, object]:
        train_cfg = self.cfg.get("training", {})
        
        if stage == "pretrain":
            lr = train_cfg.get("pretrain_lr", 1e-4)
            params = list(self.model.parameters()) + list(self.loss_fn.parameters())
            optimizer = AdamW(params, lr=lr, weight_decay=1e-2, betas=(0.9, 0.98))
            epochs = train_cfg.get("pretrain_epochs", 50)
            
        else:  # finetune
            lr = train_cfg.get("finetune_lr", 5e-5)
            # 冻结前2层（可选）
            freeze_layers = train_cfg.get("freeze_encoder_layers", 0)
            if freeze_layers > 0:
                for i, layer in enumerate(self.model.transformer.layers):
                    if i < freeze_layers:
                        for p in layer.parameters():
                            p.requires_grad = False
                logger.info(f"Frozen first {freeze_layers} transformer layers")
            
            # 分层学习率
            encoder_params = list(self.model.embedder.parameters()) + \
                             list(self.model.transformer.parameters())
            head_params = list(self.model.mlm_head.parameters()) + \
                         list(self.model.pheno_head.parameters()) + \
                         list(self.model.expr_head.parameters()) + \
                         list(self.model.causal.parameters()) + \
                         list(self.model.gene_gnn.parameters()) + \
                         list(self.loss_fn.parameters())
            
            optimizer = AdamW([
                {"params": encoder_params, "lr": lr * 0.1},   # 慢速更新编码器
                {"params": head_params, "lr": lr}              # 快速更新预测头
            ], weight_decay=1e-2)
            epochs = train_cfg.get("finetune_epochs", 100)
        
        # Warmup + Cosine Decay
        warmup_epochs = max(1, epochs // 10)
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[warmup_epochs])
        
        return optimizer, scheduler
    
    def train_epoch(self, loader, optimizer, stage: str) -> Dict[str, float]:
        self.model.train()
        total_losses = {}
        n_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch, stage=stage)
                loss, losses = self.loss_fn(outputs, batch, stage=stage)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
            
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            n_batches += 1
            
            if batch_idx % self.log_interval == 0:
                logger.info(f"  Batch {batch_idx}/{len(loader)} | "
                           f"Loss: {losses.get('total', loss.item()):.4f} | "
                           f"Pheno: {losses.get('pheno', 0):.4f}")
        
        return {k: v / n_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def eval_epoch(self, loader, stage: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.model.eval()
        total_losses = {}
        all_preds, all_trues = [], []
        n_batches = 0
        
        for batch in loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch, stage=stage)
                _, losses = self.loss_fn(outputs, batch, stage=stage)
            
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            n_batches += 1
            
            if "pheno_pred" in outputs and "pheno" in batch:
                all_preds.append(outputs["pheno_pred"].cpu())
                all_trues.append(batch["pheno"].cpu())
        
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        
        metrics = {}
        if all_preds:
            pred_cat = torch.cat(all_preds, dim=0)
            true_cat = torch.cat(all_trues, dim=0)
            metrics = compute_metrics(pred_cat, true_cat)
        
        return avg_losses, metrics
    
    def save_checkpoint(self, epoch: int, stage: str,
                        val_loss: float, tag: str = ""):
        ckpt = {
            "epoch": epoch,
            "stage": stage,
            "val_loss": val_loss,
            "model_state": self.model.state_dict(),
            "loss_fn_state": self.loss_fn.state_dict(),
            "history": self.history
        }
        fname = f"{stage}_{tag}_epoch{epoch:03d}_val{val_loss:.4f}.pt"
        path = self.checkpoint_dir / fname
        torch.save(ckpt, path)
        
        # 保存best软链接
        best_path = self.checkpoint_dir / f"{stage}_best.pt"
        torch.save(ckpt, best_path)
        
        logger.info(f"Checkpoint saved: {path}")
        return str(path)
    
    def load_checkpoint(self, path: str, strict: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(ckpt["model_state"], strict=strict)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected[:5]}...")
        if "loss_fn_state" in ckpt:
            try:
                self.loss_fn.load_state_dict(ckpt["loss_fn_state"])
            except Exception as e:
                logger.warning(f"Could not load loss_fn state: {e}")
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
        return ckpt
    
    def run_pretrain(self, train_loader, val_loader):
        """Stage 1: Masked Haplotype Modeling 预训练"""
        logger.info("=" * 60)
        logger.info("Stage 1: Masked Haplotype Modeling Pretraining")
        logger.info("=" * 60)
        
        train_cfg = self.cfg.get("training", {})
        epochs = train_cfg.get("pretrain_epochs", 50)
        optimizer, scheduler = self.build_optimizer("pretrain")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            train_losses = self.train_epoch(train_loader, optimizer, "pretrain")
            val_losses, val_metrics = self.eval_epoch(val_loader, "pretrain")
            scheduler.step()
            
            val_loss = val_losses.get("mlm", float("inf"))
            dt = time.time() - t0
            
            logger.info(
                f"[Pretrain] Epoch {epoch:3d}/{epochs} | "
                f"Train MLM: {train_losses.get('mlm', 0):.4f} | "
                f"Val MLM: {val_loss:.4f} | "
                f"Time: {dt:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}"
            )
            
            self.history["train"].append({"epoch": epoch, "stage": "pretrain", **train_losses})
            self.history["val"].append({"epoch": epoch, "stage": "pretrain", **val_losses})
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, "pretrain", val_loss, "best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info(f"Pretrain done. Best epoch: {self.best_epoch}, Val MLM: {self.best_val_loss:.4f}")
    
    def run_finetune(self, train_loader, val_loader,
                     pretrain_ckpt: Optional[str] = None):
        """Stage 2: 多任务微调"""
        logger.info("=" * 60)
        logger.info("Stage 2: Multi-task Fine-tuning")
        logger.info("=" * 60)
        
        if pretrain_ckpt:
            self.load_checkpoint(pretrain_ckpt, strict=False)
            logger.info(f"Loaded pretrain weights: {pretrain_ckpt}")
        
        train_cfg = self.cfg.get("training", {})
        epochs = train_cfg.get("finetune_epochs", 100)
        optimizer, scheduler = self.build_optimizer("finetune")
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            train_losses = self.train_epoch(train_loader, optimizer, "finetune")
            val_losses, val_metrics = self.eval_epoch(val_loader, "finetune")
            scheduler.step()
            
            # 监控指标：表型loss + 因果loss
            monitor_loss = val_losses.get("pheno", 0) + val_losses.get("causal", 0)
            
            dt = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            
            logger.info(
                f"[Finetune] Epoch {epoch:3d}/{epochs} | "
                f"Train Total: {train_losses.get('total', 0):.4f} | "
                f"Val Pheno: {val_losses.get('pheno', 0):.4f} | "
                f"R²: {val_metrics.get('r2', 0):.4f} | "
                f"Pearson: {val_metrics.get('pearson_r', 0):.4f} | "
                f"Time: {dt:.1f}s | LR: {lr:.2e}"
            )
            
            self.history["train"].append({
                "epoch": epoch, "stage": "finetune", **train_losses
            })
            self.history["val"].append({
                "epoch": epoch, "stage": "finetune",
                **val_losses, **{f"metric_{k}": v for k, v in val_metrics.items()}
            })
            
            if monitor_loss < self.best_val_loss:
                self.best_val_loss = monitor_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, "finetune", monitor_loss, "best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info(f"Finetune done. Best epoch: {self.best_epoch}, "
                   f"Best Val Loss: {self.best_val_loss:.4f}")
        
        # 保存训练历史
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")
    
    @torch.no_grad()
    def evaluate_test(self, test_loader) -> Dict[str, float]:
        """测试集最终评估"""
        logger.info("Evaluating on test set...")
        self.model.eval()
        
        all_preds, all_trues = [], []
        all_causal_preds = []
        gene_scores_list = []
        
        for batch in test_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            outputs = self.model(batch, stage="finetune")
            
            if "pheno_pred" in outputs and "pheno" in batch:
                all_preds.append(outputs["pheno_pred"].cpu())
                all_trues.append(batch["pheno"].cpu())
            if "causal_causal_pred" in outputs:
                all_causal_preds.append(outputs["causal_causal_pred"].cpu())
            if "gene_scores" in outputs:
                gene_scores_list.append(outputs["gene_scores"].cpu())
        
        results = {}
        if all_preds:
            pred_cat = torch.cat(all_preds, dim=0)
            true_cat = torch.cat(all_trues, dim=0)
            results["direct_pred"] = compute_metrics(pred_cat, true_cat)
        
        if all_causal_preds:
            cpred = torch.cat(all_causal_preds, dim=0)
            results["causal_pred"] = compute_metrics(cpred, true_cat)
        
        if gene_scores_list:
            results["mean_gene_scores"] = torch.stack(gene_scores_list).mean(0).numpy()
        
        logger.info(f"Test Results:")
        for k, v in results.items():
            if isinstance(v, dict):
                logger.info(f"  {k}: {v}")
        
        return results


# ===========================================================================
# 4. MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="MHGT-GNN Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune", "both"],
                        default="both", help="训练阶段")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="预训练checkpoint路径（finetune时使用）")
    parser.add_argument("--processed_data", type=str,
                        default="./data/processed/processed_data.pkl")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    
    # 设备
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU - training will be slow")
    
    # 加载数据
    logger.info(f"Loading processed data from {args.processed_data}")
    dataset, processed_data = load_processed_dataset(
        args.processed_data,
        mask_ratio=cfg.get("tokenize", {}).get("mask_ratio", 0.15),
        pretrain_mode=(args.stage == "pretrain")
    )
    
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    train_loader, val_loader, test_loader, splits = create_dataloaders(
        dataset, cfg, batch_size=batch_size
    )
    
    # 构建模型
    model = build_model(cfg, processed_data)
    
    # 训练器
    trainer = Trainer(model, cfg, device, args.checkpoint_dir)
    
    # 执行训练
    if args.stage in ["pretrain", "both"]:
        # 预训练使用pretrain_mode dataset
        pretrain_dataset, _ = load_processed_dataset(
            args.processed_data, mask_ratio=0.15, pretrain_mode=True
        )
        pt_train, pt_val, _, _ = create_dataloaders(pretrain_dataset, cfg, batch_size)
        trainer.run_pretrain(pt_train, pt_val)
        pretrain_ckpt = str(Path(args.checkpoint_dir) / "pretrain_best.pt")
    else:
        pretrain_ckpt = args.checkpoint
    
    if args.stage in ["finetune", "both"]:
        trainer.run_finetune(train_loader, val_loader, pretrain_ckpt)
    
    # 最终测试集评估
    best_ckpt = Path(args.checkpoint_dir) / "finetune_best.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt))
    
    test_results = trainer.evaluate_test(test_loader)
    
    # 保存测试结果
    results_path = Path(args.checkpoint_dir) / "test_results.json"
    with open(results_path, "w") as f:
        serializable = {}
        for k, v in test_results.items():
            if isinstance(v, dict):
                serializable[k] = v
            elif hasattr(v, "tolist"):
                serializable[k] = v.tolist()[:20]  # 只保存前20个基因得分
        json.dump(serializable, f, indent=2)
    
    logger.info(f"✅ Training complete. Results saved to {results_path}")


if __name__ == "__main__":
    # 允许直接import模块名
    import importlib, sys
    # 重命名以便import
    sys.modules["preprocess_01"] = importlib.import_module("01_preprocess") if False else \
        type(sys)("preprocess_01")
    main()
