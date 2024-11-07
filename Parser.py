import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ID-GRec")

    parser.add_argument("--seed_flag", type=bool, default=True, help="Fix random seed or not")

    parser.add_argument("--seed", type=int, default=2024, help="random seed for init")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    parser.add_argument("--model", type=str, default="unknown", help="model name")

    return parser.parse_args()
