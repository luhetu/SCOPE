# 延迟导入 - 只在需要时导入，避免环境依赖冲突
def build_task(args):
    if args.task == 'cls':
        from .classification import ClassificationTask
        return ClassificationTask(args)
    elif args.task == 'det':
        from .detection import DetectionTask
        return DetectionTask(args)
    elif args.task == 'seg':
        from .segmentation import SegmentationTask
        return SegmentationTask(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")
