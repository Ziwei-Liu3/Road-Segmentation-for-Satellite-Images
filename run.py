from train import train
from test import test
from mask_to_submission import submit

if __name__ == "__main__":
    print("================================================")
    print("Training Phase")
    print("================================================")
    train()

    print("================================================")
    print("Test Phase")
    print("================================================")
    test()

    print("================================================")
    print("Converting prediction to submission format")
    print("================================================")
    submit()

    print("The Dinknet152.csv is the final result for submission")
