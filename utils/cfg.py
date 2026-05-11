import yaml
import os


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
            cfg = yaml.safe_load(f)
        
        # 将 YAML Configuration直接Setup到 args 对象，确保类型正确
        for key, value in cfg.items():
            # 确保数值类型参数被正确Convert
            if key in ['lr', 'min_lr']:
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
                value = bool(value)
            elif key in ['pretrained'] and value == 'null':
                value = None
            setattr(args, key, value)
        
        print(f"✅ [cfg] SuccessLoad {len(cfg)} 个参数")
        if cli_model is not None:
            args.model = cli_model
            print(f"✅ [cfg] Override model from CLI: {cli_model}")
    elif args.cfg:
        print(f"⚠️  [cfg] Configuration文件不存在：{args.cfg}")
    
    return args
