"""
MHGT-GNN Pipeline - Step 5: Master Runner
==========================================
统一入口，串联所有步骤，支持端到端运行或单步运行

Usage (端到端):
    python run_pipeline.py --config config.yaml

Usage (分步):
    python run_pipeline.py --config config.yaml --step preprocess
    python run_pipeline.py --config config.yaml --step pretrain
    python run_pipeline.py --config config.yaml --step finetune
    python run_pipeline.py --config config.yaml --step interpret
"""

import os
import sys
import json
import time
import logging
import argparse
import pickle
import importlib.util
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)


# ===========================================================================
# MODULE LOADER (处理文件名以数字开头的情况)
# ===========================================================================

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_all_modules(base_dir):
    base = Path(base_dir)
    mods = {}
    mods["preprocess"] = load_module("preprocess_01", base / "01_preprocess.py")
    mods["model"] = load_module("model_02", base / "02_model.py")
    mods["train"] = load_module("train_03", base / "03_train.py")
    mods["interpret"] = load_module("interpret_04", base / "04_interpret.py")
    return mods


# ===========================================================================
# PIPELINE STEPS
# ===========================================================================

def step_preprocess(cfg, mods):
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*60)
    processed = mods["preprocess"].run_preprocessing(cfg)
    logger.info("✅ Preprocessing complete")
    return processed


def step_pretrain(cfg, mods, processed_data_path, device):
    import torch
    logger.info("\n" + "="*60)
    logger.info("STEP 2: SELF-SUPERVISED PRETRAINING (MLM)")
    logger.info("="*60)
    
    dataset, processed_data = mods["preprocess"].load_processed_dataset(
        processed_data_path, mask_ratio=0.15, pretrain_mode=True
    )
    
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    train_loader, val_loader, _, _ = mods["preprocess"].create_dataloaders(
        dataset, cfg, batch_size=batch_size
    )
    
    model = mods["model"].build_model(cfg, processed_data)
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir", "./checkpoints")
    
    trainer = mods["train"].Trainer(model, cfg, device, checkpoint_dir)
    trainer.run_pretrain(train_loader, val_loader)
    
    logger.info("✅ Pretraining complete")
    return str(Path(checkpoint_dir) / "pretrain_best.pt")


def step_finetune(cfg, mods, processed_data_path, device, pretrain_ckpt=None):
    import torch
    logger.info("\n" + "="*60)
    logger.info("STEP 3: MULTI-TASK FINE-TUNING")
    logger.info("="*60)
    
    dataset, processed_data = mods["preprocess"].load_processed_dataset(
        processed_data_path, mask_ratio=0.15, pretrain_mode=False
    )
    
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    train_loader, val_loader, test_loader, splits = mods["preprocess"].create_dataloaders(
        dataset, cfg, batch_size=batch_size
    )
    
    model = mods["model"].build_model(cfg, processed_data)
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir", "./checkpoints")
    
    trainer = mods["train"].Trainer(model, cfg, device, checkpoint_dir)
    trainer.run_finetune(train_loader, val_loader, pretrain_ckpt)
    
    # 测试集评估
    best_ckpt = Path(checkpoint_dir) / "finetune_best.pt"
    if best_ckpt.exists():
        import torch
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
    
    test_results = trainer.evaluate_test(test_loader)
    
    logger.info("✅ Fine-tuning complete")
    logger.info(f"   Direct Pred R²: {test_results.get('direct_pred', {}).get('r2', 'N/A'):.4f}")
    logger.info(f"   Pearson r: {test_results.get('direct_pred', {}).get('pearson_r', 'N/A'):.4f}")
    
    return str(best_ckpt), test_results


def step_interpret(cfg, mods, processed_data_path, finetune_ckpt, device):
    import torch
    logger.info("\n" + "="*60)
    logger.info("STEP 4: INTERPRETATION & GENE PRIORITIZATION")
    logger.info("="*60)
    
    dataset, processed_data = mods["preprocess"].load_processed_dataset(processed_data_path)
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    _, _, test_loader, _ = mods["preprocess"].create_dataloaders(
        dataset, cfg, batch_size=batch_size
    )
    
    model = mods["model"].build_model(cfg, processed_data)
    if finetune_ckpt and Path(finetune_ckpt).exists():
        ckpt = torch.load(finetune_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)
    
    output_dir = cfg.get("output", {}).get("results_dir", "./results/interpretation")
    results = mods["interpret"].run_interpretation(
        model, test_loader, processed_data, device, output_dir
    )
    
    # 打印训练历史图
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir", "./checkpoints")
    history_path = Path(checkpoint_dir) / "training_history.json"
    if history_path.exists():
        try:
            mods["interpret"].plot_training_history(
                str(history_path),
                str(Path(output_dir) / "training_history.png")
            )
        except Exception as e:
            logger.warning(f"Could not plot training history: {e}")
    
    logger.info("✅ Interpretation complete")
    return results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MHGT-GNN: Wheat Stripe Rust Resistance Gene Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 端到端运行
  python run_pipeline.py --config config.yaml

  # 仅预处理
  python run_pipeline.py --config config.yaml --step preprocess

  # 从现有预处理数据开始训练
  python run_pipeline.py --config config.yaml --step pretrain

  # 使用预训练权重微调
  python run_pipeline.py --config config.yaml --step finetune \\
    --pretrain_ckpt ./checkpoints/pretrain_best.pt

  # 解释已训练模型
  python run_pipeline.py --config config.yaml --step interpret \\
    --finetune_ckpt ./checkpoints/finetune_best.pt
        """
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML配置文件路径")
    parser.add_argument("--step", type=str,
                        choices=["all", "preprocess", "pretrain", "finetune", "interpret"],
                        default="all", help="运行步骤")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--finetune_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    t_start = time.time()
    
    # 加载模块
    base_dir = Path(__file__).parent
    mods = load_all_modules(base_dir)
    
    # 加载配置
    cfg = mods["preprocess"].load_config(args.config)
    
    # 设备
    import torch
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name()} | "
                   f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️  No GPU detected. Training will be slow.")
    
    processed_data_path = str(
        Path(cfg["data"]["out_dir"]) / "processed_data.pkl"
    )
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir", "./checkpoints")
    
    # 运行步骤
    pretrain_ckpt = args.pretrain_ckpt
    finetune_ckpt = args.finetune_ckpt
    
    if args.step in ["all", "preprocess"]:
        step_preprocess(cfg, mods)
    
    if args.step in ["all", "pretrain"]:
        pretrain_ckpt = step_pretrain(cfg, mods, processed_data_path, device)
    
    if args.step in ["all", "finetune"]:
        # 自动找预训练checkpoint
        if pretrain_ckpt is None:
            auto_ckpt = Path(checkpoint_dir) / "pretrain_best.pt"
            if auto_ckpt.exists():
                pretrain_ckpt = str(auto_ckpt)
                logger.info(f"Auto-found pretrain checkpoint: {pretrain_ckpt}")
        finetune_ckpt, test_results = step_finetune(
            cfg, mods, processed_data_path, device, pretrain_ckpt
        )
    
    if args.step in ["all", "interpret"]:
        if finetune_ckpt is None:
            auto_ckpt = Path(checkpoint_dir) / "finetune_best.pt"
            if auto_ckpt.exists():
                finetune_ckpt = str(auto_ckpt)
        step_interpret(cfg, mods, processed_data_path, finetune_ckpt, device)
    
    total_time = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Pipeline complete in {total_time/3600:.2f} hours")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
