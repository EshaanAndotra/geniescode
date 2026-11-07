"""Reproduce the audit, historical replay, policy portfolio and final files."""
from pathlib import Path
import argparse,subprocess,sys

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--refresh',action='store_true');args=ap.parse_args()
    jobs=[['audit_phase2.py'],['replay_v2.py']+(['--refresh'] if args.refresh else []),
          ['evaluate_policy_v2.py'],['select_policy_portfolio.py'],['finalize_v2.py'],['build_phase2_upload.py'],['-m','unittest','test_phase2','-v']]
    for job in jobs:
        print('\nRunning '+' '.join(job),flush=True)
        subprocess.run([sys.executable,*job],cwd=ROOT,check=True)

if __name__=='__main__':main()
