import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Data name', default='kiba')
    parser.add_argument('--split_mode', type=str, help='Data split', default='S1')
    parser.add_argument('--cuda_id', type=int, help='Cuda id', default=0)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs', default=8000)  # davis 8k, kiba 12k
    parser.add_argument('--batch_size', type=int, help='Batch size of train test', default=4096) # davis: 1024, kiba: 4096
    parser.add_argument('--lr', type=float, help='Learning rate to train', default=0.0005)
    parser.add_argument('--fold', type=int, help='Fold of 5-CV', default=-100)
    parser.add_argument('--dropedge_rate', type=float, help='Edge dropout rate', default=0.2)
    parser.add_argument('--skip', type=bool, help='Whether use skip connection', default=False)
    parser.add_argument('--ratio', type=float, help='edge density', default=0.0115) # davis 5%, kiba 1.15%
    parser.add_argument('--sim_d', type=int, help='', default=5) # S2: davis5, kiba4, S3
    parser.add_argument('--sim_t', type=int, help='', default=10)
    parser.add_argument('--refine_epoch', help='Which epoch to refine adjacency matrix', type=lambda s: list(map(int, s.split(','))), default=[]) # davis: [2000, 4000]; kiba: [4000, 6000]
    args = parser.parse_args()

    return args