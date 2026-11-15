import argparse
from pathlib import Path
import pandas as pd
import model as M
import router

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True);p.add_argument('--image_root',type=Path,required=True)
p.add_argument('--model_dir',type=Path,default=Path('.'));p.add_argument('--output',type=Path,default=Path('submission.csv'))
p.add_argument('--sample_submission',type=Path)
a=p.parse_args();f=pd.read_csv(a.input);assert f.sample_id.is_unique
_,pred,_=router.infer(f,M.image_features(f,a.image_root),router.load(a.model_dir))
ids=pd.read_csv(a.sample_submission).sample_id if a.sample_submission else f.sample_id
result=pd.Series(pred,index=f.sample_id);assert ids.is_unique and set(ids)==set(result.index)
pd.DataFrame({'sample_id':ids,'label':result.loc[ids].to_numpy()}).to_csv(a.output,index=False)
print('Saved',a.output,len(ids))
