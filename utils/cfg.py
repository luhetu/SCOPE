import yaml
import os


_NULL_STRINGS = {"", "none", "null"}


def _normalize_cfg_value(value):
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        return None
    return value


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


def _convert_cfg_value(key, value):
    value = _normalize_cfg_value(value)
    if value is None:
        return None
    if key in ['lr', 'min_lr']:
        return float(value)
    if key in [
        'bs', 'size', 'n_epochs', 'max_iters', 'warmup_iters',
        'checkpoint_interval', 'eval_interval', 'log_interval',
        'patch', 'dim', 'depth', 'heads', 'mlp_dim', 'dim_head',
        'seg_head_dim', 'seg_aux_dim', 'seg_neck_dim'
    ]:
        return int(value)
    if key in [
        'warmup_epochs', 'drop_path_rate', 'weight_decay',
        'dropout', 'emb_dropout', 'layer_decay_rate'
    ]:
        return float(value)
    if key in ['amp', 'aug', 'nowandb', 'use_cls_token']:
        return _as_bool(value)
    return value


def load_cfg(parser):
    """
    ⚙️ Load YAML Configuration文件并合并到 args
    """
    # 先解析命令行参数
    args = parser.parse_args()
    cli_model = getattr(args, 'model', None)
    
    # 如果指定了Configuration文件，Load并覆盖
    if args.cfg and os.path.isfile(args.cfg):
        print(f"✅ [cfg] LoadConfiguration文件：{args.cfg}")
        with open(args.cfg, "r", encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Configuration must be a mapping: {args.cfg}")
        
        # 将 YAML Configuration直接Setup到 args 对象，确保类型正确
        for key, value in cfg.items():
            setattr(args, key, _convert_cfg_value(key, value))
        
        print(f"✅ [cfg] SuccessLoad {len(cfg)} 个参数")
        if cli_model is not None:
            args.model = cli_model
            print(f"✅ [cfg] Override model from CLI: {cli_model}")
    elif args.cfg:
        print(f"⚠️  [cfg] Configuration文件不存在：{args.cfg}")
    
    return args
