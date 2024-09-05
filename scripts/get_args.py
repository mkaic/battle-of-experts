from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-p", "--only_print_params", action="store_true", default=False)
    parser.add_argument("-c", "--ckpt", type=str, default=None)
    parser.add_argument("-l", "--log", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-m", "--model", type=str, default="boe")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_aug", action="store_true", default=False)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-s", "--save", action="store_true", default=False)
    return parser.parse_args()
