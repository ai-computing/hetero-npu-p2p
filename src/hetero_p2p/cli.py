import argparse,json
from .backend import probe

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['doctor']);p.add_argument('--device',default='/dev/rngd/npu1ch0');a=p.parse_args()
    result=probe(a.device);print(json.dumps(result,indent=2))
    if not result['supported']:raise SystemExit(2)
