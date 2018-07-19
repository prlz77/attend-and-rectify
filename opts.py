import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("root_folder", type=str, help="Dataset root folder")
parser.add_argument("train_listfile", type=str, help="Train dataset listfile.")
parser.add_argument("val_listfile", type=str, help="Val dataset listfile.")
parser.add_argument("nlabels", type=int, help="Number of output labels")

# Optimization
parser.add_argument("--lr", "-l", type=float, default=0.1, help="Learning rate.")
parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs.")
parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size.")
parser.add_argument("--schedule", type=int, nargs='*', default=[])
parser.add_argument("--test_only", action="store_true", help="Perform test step alone.")

# I/O
parser.add_argument("--log_path", type=str, default="./logs", help="Path to save logs")
parser.add_argument("--save", action="store_true", help="Whether to save model.")
parser.add_argument("--size", type=int, default=256, help="original image size")
parser.add_argument("--tencrop", action='store_true', help="whether to perform ten crop")

# Performance
parser.add_argument("--num_workers", type=int, default=3, help="Number of prefetching threads")
parser.add_argument("--ngpu", type=int, default=1, help="Number of gpus to use")

# Attention specific
parser.add_argument("--attention_depth", "-d", type=int, default=0, help="Attention depth")
parser.add_argument("--attention_width", "-w", type=int, default=0, help="Attention width")
parser.add_argument("--attention_output", "-o", type=str, default='all', help="Attention to pay at the outputs")
parser.add_argument("--attention_type", "-t", type=str, default='sigmoid', help="Attention function")
parser.add_argument('--reg_weight', '-r', type=float, default=0., help="Weight to regularize attention masks.")
parser.add_argument("--has_gates", "-g", action="store_true", help="Whether to add gating functions.")

