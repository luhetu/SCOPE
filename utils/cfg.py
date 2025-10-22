import yaml, os

def load_cfg(parser):
    args, rest = parser.parse_known_args()
    if args.cfg and os.path.isfile(args.cfg):
        with open(args.cfg, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            parser.add_argument(f'--{k}', type=type(v), default=v)
    return parser.parse_args(rest)
