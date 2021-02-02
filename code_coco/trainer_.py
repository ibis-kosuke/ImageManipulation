from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2, build_images
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, EncDecNet, VGG16
from datasets_ import prepare_data
from model import RNN_ENCODER
from VGGFeatureLoss import VGGNet

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss

import os
import time
import numpy as np
import sys
import pickle as pkl

class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, args):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            #mkdir_p(self.model_dir)
            #mkdir_p(self.image_dir)

        self.gpus = list(map(int, cfg.GPU_ID.split(',')))
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.args = args
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = 1
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.display_interval = 1000

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        ################### Text and Image encoders ######################################## 
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        """
        image_encoder = CNN_dummy()
        image_encoder.cuda()
        image_encoder.eval()
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()
        """

        VGG = VGG16()
        VGG.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        ####################### Generator and Discriminators ############## 
        from model import D_NET256
        netD = D_NET256()
        netG = EncDecNet()

        netD.apply(weights_init)
        netG.apply(weights_init)

        #
        epoch = 0
        """
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        """
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            netG.cuda()
            netD.cuda()
            VGG.cuda()
        return [text_encoder, netG, netD, epoch, VGG]

    def define_optimizers(self, netG, netD):
        optimizerD = optim.Adam(netD.parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizerD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netD, epoch):
        if not os.path.isdir(self.model_dir):
            mkdir_p(self.model_dir)
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        torch.save(netD.state_dict(), 
                '%s/netD.pth' % (self.model_dir))

        print('Save G/Ds models.')


    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires


    def save_img_results(self, netG, noise, w_sent_emb, w_words_embs,
                          caption, w_caption, gen_iterations, real_imgs, mask, VGG):

        if not os.path.isdir(self.image_dir):
            mkdir_p(self.image_dir)
        # Save images
        fake_img, _, _ = nn.parallel.data_parallel(netG, (real_imgs[-1], w_sent_emb, w_words_embs, noise, mask, VGG), self.gpus)

        img_set = build_images(real_imgs[-1], fake_img, caption, w_caption, self.ixtoword)
        fullpath = '%s/average_%d.png' % (self.image_dir, gen_iterations)
        img = Image.fromarray(img_set)
        img.save(fullpath)
        

        """
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
        """

        '''
        # save the real images 
        for k in range(8):
            im = real_imgs[-1][k].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            fullpath = '%s/R_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, k)
            im.save(fullpath)
        '''

    def train(self):
        text_encoder,netG, netD, start_epoch, VGG = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizerD = self.define_optimizers(netG, netD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        #batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        #noise = Variable(torch.FloatTensor(batch_size, nz))
        #fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        #if cfg.CUDA:
        #    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            all_sent_embs = []
            all_captions =[]
            all_keys = []
            while step < self.num_batches:
                if(step%self.display_interval==0):
                    print('step: %d' %step)
                step += 1
                gen_iterations += 1

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, full_caps, class_ids, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id = prepare_data(data)
                
                batch_size = captions.shape[0]

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef

                # matched text embeddings
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                sent_emb = sent_emb.cpu()
                full_caps = full_caps.cpu()
                all_sent_embs.append(sent_emb)
                all_captions.append(full_caps)
                all_keys.extend(keys)
            all_sent_embs = torch.cat(all_sent_embs, dim=0)
            all_captions = torch.cat(all_captions, dim=0)




            """
                # mismatched text embeddings
                w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                w_words_embs, w_sent_emb = w_words_embs.detach(), w_sent_emb.detach()

                # image features: regional and global
                #region_features, cnn_code = image_encoder(imgs[len(netsD)-1])

                mask = (wrong_caps == 0) ## 
                num_words = w_words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Modify real images
                ######################################################
                noise.data.normal_(0, 1)
                fake_img, mu, logvar = nn.parallel.data_parallel(netG, (imgs[-1], w_sent_emb, w_words_embs, noise, mask, VGG), self.gpus)

                #######################################################
                # (3) Update D network
                ######################################################

                netD.zero_grad()
                errD, D_logs = discriminator_loss(netD, imgs[-1], fake_img, sent_emb, w_sent_emb, 
                                                    real_labels, fake_labels)
                errD.backward()
                optimizerD.step()
            

                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels,
                                              words_embs, cap_lens, image_encoder, class_ids, w_words_embs, 
                                              wrong_caps_len, wrong_cls_id)
                    # backward and update parameters
                    errD.backward(retain_graph=True)
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD)


                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G

                step += 1
                gen_iterations += 1

                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netD, fake_img, imgs[-1], real_labels, w_sent_emb)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()

                #self.save_img_results(netG, fixed_noise, w_sent_emb,
                #                          w_words_embs, captions, wrong_caps, epoch, imgs)
                #self.save_model(netG, avg_param_G, netD, epoch)

                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, w_sent_emb,
                                          w_words_embs, captions, wrong_caps, epoch, imgs, mask, VGG)
                    load_params(netG, backup_para)
            """

            end_t = time.time()

            print('''[%d/%d][%d]
                  Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                         end_t - start_t))
            file_path = os.path.join(cfg.DATA_DIR, 'sent_embs.pkl')
            with open(file_path, 'wb') as f:
                pkl.dump([all_sent_embs, all_captions, all_keys, self.ixtoword], f)
            print('save sent_embs to %s' % file_path)
            
            #if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0: 
            #    self.save_model(netG, avg_param_G, netD, epoch)

        #self.save_model(netG, avg_param_G, netD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '' or cfg.TRAIN.NET_C == '':
            print('Error: the path for main module or DCM is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'

            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            # The text encoder
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()
            # The image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()

            # The VGG network
            VGG = VGGNet()
            print("Load the VGG model")
            VGG.cuda()
            VGG.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            # The DCM
            netDCM = DCM_Net()
            if cfg.TRAIN.NET_C != '':
                state_dict = \
                    torch.load(cfg.TRAIN.NET_C, map_location=lambda storage, loc: storage)
                netDCM.load_state_dict(state_dict)
                print('Load DCM from: ', cfg.TRAIN.NET_C)
            netDCM.cuda()
            netDCM.eval()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save modified images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            idx = 0 
            for _ in range(5):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)

                    imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                                wrong_caps_len, wrong_cls_id = prepare_data(data)

                    #######################################################
                    # (1) Extract text and image embeddings
                    ######################################################

                    hidden = text_encoder.init_hidden(batch_size)

                    words_embs, sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                    mask = (wrong_caps == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    region_features, cnn_code = \
                                    image_encoder(imgs[cfg.TREE.BRANCH_NUM - 1])

                    #######################################################
                    # (2) Modify real images
                    ######################################################

                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, mu, logvar, h_code, c_code = netG(noise,
                                     sent_emb, words_embs, mask, cnn_code, region_features)
                    
                    real_img = imgs[cfg.TREE.BRANCH_NUM - 1]
                    real_features = VGG(real_img)[0]

                    fake_img = netDCM(h_code, real_features, sent_emb, words_embs,\
                                         mask, c_code)
                    for j in range(batch_size):
                        s_tmp = '%s/single' % (save_dir)
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        im = fake_img[j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, idx)
                        idx = idx+1
                        im.save(fullpath)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '' or cfg.TRAIN.NET_C == '':
            print('Error: the path for main module or DCM is not found!')
        else:
            # The text encoder
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # The image encoder
            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()
            
            """
            image_encoder = CNN_dummy()
            image_encoder = image_encoder.cuda()
            image_encoder.eval()
            """

            # The VGG network
            VGG = VGGNet()
            print("Load the VGG model")
            VGG.cuda()
            VGG.eval()

            # The main module
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            #netG = nn.DataParallel(netG, device_ids= self.gpus)
            netG.cuda()
            netG.eval()

            # The DCM
            netDCM = DCM_Net()
            if cfg.TRAIN.NET_C != '':
                state_dict = \
                    torch.load(cfg.TRAIN.NET_C, map_location=lambda storage, loc: storage)
                netDCM.load_state_dict(state_dict)
                print('Load DCM from: ', cfg.TRAIN.NET_C)
            netDCM.cuda()
            netDCM.eval()

            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices, imgs = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()

                    #######################################################
                    # (1) Extract text and image embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    
                    # The text embeddings
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)

                    # The image embeddings
                    region_features, cnn_code = \
                                    image_encoder(imgs[cfg.TREE.BRANCH_NUM - 1].unsqueeze(0))
                    mask = (captions == 0)

                    #######################################################
                    # (2) Modify real images
                    ######################################################
                    noise.data.normal_(0, 1)

                    fake_imgs, attention_maps, mu, logvar, h_code, c_code =nn.parallel.data_parallel( netG, (noise,
                                     sent_emb, words_embs, mask, cnn_code, region_features), self.gpus)
                    print('fake_imgs: ', fake_imgs[0].shape)
                    print('h_code: ', h_code.shape)
                    
                    real_img = imgs[cfg.TREE.BRANCH_NUM - 1].unsqueeze(0)
                    real_features = VGG(real_img)[0]

                    ###応急処置　並列化ができない。
                    real_features = real_features[:5,:,:,:] ###
                    sent_emb = sent_emb[:5,:]
                    words_embs = words_embs[:5,:,:]
                    mask = mask[:5,:]
                    ###

                    fake_img = netDCM(h_code, real_features, sent_emb, words_embs,\
                                         mask, c_code)

                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(5): ## batch_size
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            im = np.transpose(im, (1, 2, 0))
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            """
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
                            """


                        save_name = '%s/%d_sf_%d' % (save_dir, 1, sorted_indices[j])
                        im = fake_img[j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_SF.png' % (save_name)
                        im.save(fullpath)

                    save_name = '%s/%d_s_%d' % (save_dir, 1, 9)
                    im = imgs[2].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_SR.png' % (save_name)
                    im.save(fullpath)

