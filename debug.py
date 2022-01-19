from utils import *


# root = "D:/Data/Resnet/ice-water"
# dirs = os.listdir(root)
# for idx, dir_name in enumerate(dirs):
#     relabel_train_val_from_sip_ice_water(os.path.join(root,dir_name))


with open('test.txt','wt',encoding="utf-8") as f:
    f.write('Epoch: 1\n')
    f.write('{:-^150s}\n'.format('Split line'))

with open('test.txt','a+',encoding="utf-8") as f:
    f.write('Epoch: 2')

with open('test.txt','r+',encoding="utf-8") as f:
        content = f.read()
        f.seek(0, 0)
        f.write('Epoch: 0' + '\n' + content)
  


print("Done")


