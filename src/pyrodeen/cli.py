import sys
import argparse

cli_test_message = "Command line interface works!"

def parse(arguments = []):
    parser = argparse.ArgumentParser(description="PyRoDeEn CLI")
    parser.add_argument('-t', '--test_cli', help='Test grid object', action='store_true')

    return parser.parse_args(arguments)

def main():
    args = parse(sys.argv[1:])

    if args.test_cli:
        print("Command line interface works!")

if __name__ == "__main__":
    main()