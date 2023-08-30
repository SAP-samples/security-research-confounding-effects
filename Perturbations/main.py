import os
import random
import subprocess
import argparse
import csv
from tqdm import tqdm
styles = ["LLVM", "GNU", "Google", "Chromium", "Mozilla", "WebKit", "Microsoft"]


def get_random_cpp(path):
    return random.choice(os.listdir(path))


def apply_codestyle_LLVM(file):
    stream = os.popen('/usr/local/opt/llvm/bin/clang-format --style LLVM {}'.format(file))
    output = stream.read()
    return output

def apply_codestyle_GNU(file):
    stream = os.popen('/usr/local/opt/llvm/bin/clang-format --style GNU {}'.format(file))
    output = stream.read()
    return output

def apply_codestyle_Google(file):
    stream = os.popen('/usr/local/opt/llvm/bin/clang-format --style Google {}'.format(file))
    output = stream.read()
    return output

def apply_codestyle_Chromium(file):
    stream = os.popen('/usr/local/opt/llvm/bin/clang-format --style Chromium {}'.format(file))
    output = stream.read()
    return output

def apply_codestyle_Mozilla(file):
    stream = os.popen('/usr/local/opt/llvm/bin/clang-format --style Mozilla {}'.format(file))
    output = stream.read()
    return output


def apply_py_obfuscator(file):
    stream = os.popen('/usr/local/bin/python3 src/obfuscator.py {}'.format(file))
    output = stream.read()
    return output


def apply_cobfuscate(file):
    stream = os.popen('/usr/local/bin/python3 src/cobfuscator.py {}'.format(file))
    output = stream.read().replace(file, "")
    return output

def obfuscate_then_style(file):
    stream = os.popen('/usr/local/bin/python3 src/cobfuscator.py {}'.format(file))
    output = stream.read().replace(file, "")
    fh = open("tmp", "w+")
    fh.write(output)
    fh.close()
    stream = os.popen('/usr/local/opt/llvm/bin/clang-format --style {} {}'.format(random.choice(styles), "tmp"))
    output = stream.read()
    os.remove("tmp")
    return output

def py_obfuscate_then_style(file):
    stream = os.popen('/usr/local/bin/python3 src/obfuscator.py {}'.format(file))
    output = stream.read().replace(file, "")
    fh = open("tmp", "w+")
    fh.write(output)
    fh.close()
    stream = os.popen('/usr/local/opt/llvm/bin/clang-format --style {} {}'.format(random.choice(styles), "tmp"))
    output = stream.read()
    return output

def double_obfuscate(file):
    stream = os.popen('/usr/local/bin/python3 src/cobfuscator.py {}'.format(file))
    output = stream.read().replace(file, "")
    fh = open("tmp", "w+")
    fh.write(output)
    fh.close()
    stream = os.popen('/usr/local/bin/python3 src/obfuscator.py {}'.format("tmp"))
    output = stream.read().replace(file, "")
    return output


mutators = [double_obfuscate, py_obfuscate_then_style, obfuscate_then_style, apply_codestyle_LLVM, apply_codestyle_GNU, apply_codestyle_Google, apply_codestyle_Chromium, apply_codestyle_Mozilla, apply_py_obfuscator, apply_cobfuscate]
csv.field_size_limit(1000000000)

def process(content, name, path):
    fh = open("ftmp", "w+")
    fh.write(content)
    fh.close()
    for m in mutators:
        savepath = path + "/" + m.__name__
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        out = m("ftmp")
        fh = open(savepath + "/" + name, "w+")
        fh.write(out)
        fh.close()

def justwriteout(content, name, path):
    fh = open(path + "/" + name, "w+")
    fh.write(content)
    fh.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("obfuscated dataset generator")
    parser.add_argument("csv", help="Dataset csv file.", type=str)
    parser.add_argument("output", help="Output dir", type=str)
    parser.add_argument("obfuscate", help="Obfuscate yes or no", type=str)
    args = parser.parse_args()
    output = args.output
    file = args.csv
    obfuscate = args.obfuscate
    print(output, file, obfuscate)
    with open(file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader):
                processed = row["processed_func"]
                if obfuscate == "no":
                    justwriteout(processed, row["index"] + "_" + row["target"] + ".c", output)
                else:
                    process(processed, row["index"] + "_" + row["target"]+".c" , output)

