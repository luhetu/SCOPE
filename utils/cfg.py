import yaml
import os


def load_cfg(parser):
    """
    ⚙️ 加载 YAML 配置文件并合并到 args
    """
    # 先解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了配置文件，加载并覆盖
    if args.cfg and os.path.isfile(args.cfg):
        print(f"✅ [cfg] 加载配置文件：{args.cfg}")
        with open(args.cfg, "r") as f:
            cfg = yaml.safe_load(f)
        
        # 将 YAML 配置直接设置到 args 对象，确保类型正确
        for key, value in cfg.items():
            # 确保数值类型参数被正确转换
            if key in ['lr', 'min_lr']:
                value = float(value)
            elif key in ['bs', 'size', 'n_epochs', 'patch', 'dim', 'depth', 'heads', 'mlp_dim', 'warmup_epochs']:
                value = int(value)
            elif key in ['amp', 'aug', 'nowandb']:
                value = bool(value)
            setattr(args, key, value)
        
        print(f"✅ [cfg] 成功加载 {len(cfg)} 个参数")
    elif args.cfg:
        print(f"⚠️  [cfg] 配置文件不存在：{args.cfg}")
    
    return args
