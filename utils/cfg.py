import yaml
import os


def _is_null_like(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"})


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def load_cfg(parser):
    """
    ⚙️ Load YAML Configuration文件并合并到 args
    """
    # 先解析命令行参数
    args = parser.parse_args()
    cli_overrides = {
        key: getattr(args, key, None)
        for key in ('model', 'data_dir', 'workers_per_gpu')
        if getattr(args, key, None) is not None
    }
    
    # 如果指定了Configuration文件，Load并覆盖
    if args.cfg and os.path.isfile(args.cfg):
        print(f"✅ [cfg] LoadConfiguration文件：{args.cfg}")
        with open(args.cfg, "r", encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"YAML config must be a mapping, got {type(cfg).__name__}: {args.cfg}")
        
        # 将 YAML Configuration直接Setup到 args 对象，确保类型正确
        for key, value in cfg.items():
            if _is_null_like(value):
                value = None
            # 确保数值类型参数被正确Convert
            if value is None:
                pass
            elif key in ['lr', 'min_lr']:
                value = float(value)
            elif key in [
                'bs', 'size', 'n_epochs', 'max_iters', 'warmup_iters',
                'checkpoint_interval', 'eval_interval', 'log_interval',
                'patch', 'dim', 'depth', 'heads', 'mlp_dim', 'dim_head',
                'seg_head_dim', 'seg_aux_dim', 'seg_neck_dim'
            ]:
                value = int(value)
            elif key in [
                'warmup_epochs', 'drop_path_rate', 'weight_decay',
                'dropout', 'emb_dropout', 'layer_decay_rate'
            ]:
                value = float(value)
            elif key in ['amp', 'aug', 'nowandb', 'use_cls_token']:
                value = _as_bool(value)
            setattr(args, key, value)
        
        print(f"✅ [cfg] SuccessLoad {len(cfg)} 个参数")
        for key, value in cli_overrides.items():
            setattr(args, key, value)
            print(f"✅ [cfg] Override {key} from CLI: {value}")
    elif args.cfg:
        print(f"⚠️  [cfg] Configuration文件不存在：{args.cfg}")
    
    return args
