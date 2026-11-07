"""Package code, source data, submissions and evidence without bulky caches."""
from pathlib import Path
import hashlib,json,zipfile
import pandas as pd
from data import load_data,validate_submission

ROOT=Path(__file__).resolve().parent

def main():
    data=load_data(ROOT/'inputs_phase2')
    for name in ['submission_final.csv','submission_challenger.csv']:
        validate_submission(pd.read_csv(ROOT/'outputs_phase2'/name),data)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'requirements.txt']
    for folder in ['inputs','inputs_phase2','outputs','outputs_phase2']:
        files.extend(p for p in (ROOT/folder).iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md','.png'])
    files=sorted(set(files));destination=ROOT.parent/'halcyon_final_stage_model.zip';prefix=Path('halcyon_phase2')
    manifest={}
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in files:
            rel=file.relative_to(ROOT)
            if rel==Path('README.md'):rel=Path('README_PHASE1.md')
            elif rel==Path('README_PHASE2.md'):rel=Path('README.md')
            z.write(file,prefix/rel);manifest[str(rel)]=hashlib.sha256(file.read_bytes()).hexdigest()
        z.writestr(str(prefix/'package_manifest.json'),json.dumps(manifest,indent=2))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for name in ['submission_final.csv','submission_challenger.csv']:
            assert z.read(str(prefix/'outputs_phase2'/name).replace('\\','/'))==(ROOT/'outputs_phase2'/name).read_bytes()
        print(json.dumps({'archive':str(destination),'bytes':destination.stat().st_size,'files':len(z.namelist())},indent=2))

if __name__=='__main__':main()
