
import argparse
import os

parser = argparse.ArgumentParser()

NUM_WORKERS = os.cpu_count()

parser.add_argument(
    "-t", "--train_dir", type=str, required=True, help="Path to training data."
)

parser.add_argument(
    "-e", "--test_dir", type=str, required=True, help="Path to testing data."
)

parser.add_argument(
    "-b", "--batch_size", type=int, default=32, help="Number of samples per batch."
)

parser.add_argument(
    "-w", "--num_workers", type=int, default=NUM_WORKERS, help="Number of workers per DataLoader."
)

print(parser.parse_args())
