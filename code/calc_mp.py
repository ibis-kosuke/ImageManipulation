import os
import sys
import glob
import subprocess
from subprocess import PIPE

args = sys.argv ##args[1]: netG, [2]: gpu_id

netG_path = '/data/unagi0/ktokitake/encdecmodel/birds/output/%s/Model/*.pth' % args[1]
netG_epochs = glob.glob(netG_path)

netG_epochs_new = []
for f in netG_epochs:
    f = os.path.basename(f)
    if f=='netD.pth':
        continue
    netG_epochs_new.append(f)

log_path = '/data/unagi0/ktokitake/encdecmodel/birds/output/%s/log_file' % args[1]
logs = open(log_path, "w")

for i, netG_epoch in enumerate(netG_epochs_new):
    print('now_processing: %s' % netG_epoch)
    proc = subprocess.run('python main.py --cfg cfg/eval_bird.yml --gpu %s --netG %s --netG_epoch %s' 
                            %(args[2], args[1], netG_epoch), shell=True, stdout=PIPE, stderr=PIPE)                            
    print(proc.stdout.decode('utf-8'))
    print(proc.stderr.decode('utf-8'))
    std_out = proc.stdout.decode('utf-8').split('\n')
    logs.write(std_out[-3]+'\n')

logs.close()



