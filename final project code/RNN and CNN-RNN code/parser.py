import argparse

def checkParser():
    parser = argparse.ArgumentParser(description="Configure the network")
    parser.add_argument("--model", "-m", dest="model_type", type=str, default="RNN", help="type of model (RNN, LSTM, GRU)")
    parser.add_argument("--prep", "-p", dest="use_prep", action="store_true", default=False, help="use data preperation or not")
    parser.add_argument("--epochs", "-e", dest="num_epochs", type=int, default=50, help="number of the epochs")
    parser.add_argument("--dropout", "-d", dest="dropout", type=float, default=0.6, help="dropout value")
    parser.add_argument("--subject", "-s", dest="use_subject", action="store_true", default=False, help="use single person data")
    parser.add_argument("--window", "-w", dest="window", type=int, default=None, help="window size")
    
    return parser.parse_args()


if __name__=="__main__":
    args = checkParser()
    print(args.model_type)
    print(args.use_prep)
