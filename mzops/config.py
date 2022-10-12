from mzops.utils import artifact_dir
import os

d={}
d['status'] ='predict'
d['sub_data'] = False
d['sample'] = 'how to make  html more dynamique by php and jquery and connect to mysql '

d['artifactdir'] = artifact_dir()
d['model_artifact'] = os.path.join(artifact_dir(),'model.gz')
