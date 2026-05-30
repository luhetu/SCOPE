import yaml
import os
import sys


def _get_explicit_cli_dests(parser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    option_to_dest = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    for token in argv:
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def load_cfg(parser):
    """
    ⚙️ Load YAML Configuration文件并合并到 args
    """
    # 先解析命令行参数
    explicit_cli_dests = _get_explicit_cli_dests(parser)
    args = parser.parse_args()
    cli_overrides = {key: getattr(args, key) for key in explicit_cli_dests if key != 'cfg' and hasattr(args, key)}
    
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
        for key, value in cli_overrides.items():
            setattr(args, key, value)
            print(f"✅ [cfg] Override {key} from CLI: {value}")
    elif args.cfg:
        print(f"⚠️  [cfg] Configuration文件不存在：{args.cfg}")
    
    return args
