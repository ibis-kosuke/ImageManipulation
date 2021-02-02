import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.inception import inception_v3

import argparse
import os
import pickle as pkl
import numpy as np
from PIL import Image
import glob

def get_imgs(files, args, transform=None):
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        #32x32x3
        if transform is not None:
            img = transform(img)
            #3x299x299, -1~1
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    if args.cuda:
        imgs = imgs.cuda()
    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="birds_attn2_2020_09_16_18_17_18" ,help="dirs split by , ")
    parser.add_argument('--data_dir', type=str, default="/data/unagi0/ktokitake/encdecmodel/birds") #cifar10
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
   
    args = parser.parse_args()

    if args.gpu_id !=-1:
        args.cuda =True
        torch.cuda.set_device(args.gpu_id)

    dirs_list = args.dir.split(",")

    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    
    print("loading inception_v3 model...")
    model = inception_v3(pretrained=True, transform_input=False)
    if args.cuda:
        model.cuda()
    model.eval()
        
    print("getting imgs")
    #imgs = get_imgs_from_dir(args.dir, args, transform)
    #imgs: num_all_sample x 3 x 299 x 299
    valid_dirs_dic = {}
    for netG_path in dirs_list:
        all_file = []
        for i in range(5):
            valid_dir = os.path.join(args.data_dir, 'output', netG_path, 'valid/valid_%d/single' % i)
            all_file.extend(glob.glob(valid_dir+'/*.png'))
        valid_dirs_dic[netG_path] = all_file

    """
    dirs = os.listdir(valid_dirs[0])
    valid_dirs_dic = {}
    for valid_dr in valid_dirs:
        all_file = []
        for dr in dirs:
            files = glob.glob(os.path.join(valid_dr,dr,"*.png"))
            all_file.extend(files)
        #num_samples = len(all_file)
        valid_dirs_dic[valid_dr] = all_file
    """
    """
    else:
        cifar10_dir = "/data/unagi0/ktokitake/cifar10/cifar-10-batches-py"
        all_file = glob.glob(cifar10_dir+"/data_batch_*")
        all_imgs = []
        for f in all_file:
            with open(f, "rb") as g:
                data_dic = pkl.load(g, encoding="latin1")
                imgs = data_dic['data']
                imgs = imgs.reshape(-1,3,32,32)
                imgs = np.transpose(imgs, (0,2,3,1))
                ##imgs: 10000 x 32 x 32 x 3
                all_imgs.append(imgs)
        all_imgs = np.concatenate(all_imgs, axis=0).astype(np.uint8)
        assert all_imgs.shape == (5*10000, 32, 32, 3)
        num_samples = all_imgs.shape[0]
    """
        

    for netG_path in dirs_list:
        batch_size = args.batch_size
        dis_interval = 50
        all_file = valid_dirs_dic[netG_path]
        num_samples = len(all_file)
        num_batches = (num_samples+batch_size-1 ) // batch_size
        all_score = []
        for i in range(num_batches):
            if (i+1) % dis_interval == 0:
                print("now processing: {}/{}".format(i+1, num_batches))

            imgs = get_imgs(all_file[i*batch_size:(i+1)*batch_size], args, transform)
            preds = model(imgs)
            preds = F.softmax(preds)
            preds = preds.detach().cpu().numpy()
            py = np.mean(preds, axis=0)
            e = preds*np.log(preds/py)
            kl_div = np.sum(e, axis=1)
            score = np.exp(np.mean(kl_div, axis=0))
            all_score.append(score)
        print("incepton score for {},  mean:{}, std:{}".format(netG_path, np.mean(all_score), np.std(all_score)))


    '''
    batch_size = args.batch_size
    dis_interval = 50
    num_batches = (imgs.shape[0]+batch_size-1 ) // args.batch_size
    
    all_preds = []
    for i in range(num_batches):
        if (i+1) % dis_interval == 0:
            print("now processing: {}/{}".format(i+1, num_batches))
        batch_imgs = imgs[i*batch_size:(i+1)*batch_size, :,:,:]
        #preds = torch.randn(10,300)
        preds = model(batch_imgs)
        preds = F.softmax(preds)
        #preds: batch x class_num(?)
        #all_preds.append(preds)
        print("preds shape: {}".format(preds.shape))




    all_preds = torch.cat(all_preds, dim=0)
    #num_all_sample x class_num
    all_preds = all_preds.numpy()
    py = np.mean(all_preds, axis=0)
    e = all_preds * np.log(all_preds/py)
    kl_div = np.sum(e, axis=1)
    ans = np.mean(kl_div)
    inception_score = np.exp(ans)
    
    print("inception score: {:.10f}".format(inception_score))
    '''









