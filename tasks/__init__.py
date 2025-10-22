from .classification import ClassificationTask

def build_task(args):
    if args.task == 'cls':
        return ClassificationTask(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")
