import os, sys

TEST_DIR = "tests"

args = sys.argv[1:]

files = os.listdir(TEST_DIR)

if len(args) > 0:
    files = [x for x in files if any([a in x for a in args])]

if len(files) == 0:
    print("No tests found")
    exit(1)

print("Running the following tests:\n", '\n'.join(files), '\n')

for test in files:
    filename = f"{TEST_DIR}/{test}"
    if os.path.isfile(filename) and test != "__init__.py":
        print(f"=== [{test}] ===")
        __import__(filename.replace("/", ".").replace(".py", ""), globals(), locals())

