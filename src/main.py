from helper import parameter_parser
from walklets import WalkletMachine

def main(args):
    WalkletMachine(args)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
