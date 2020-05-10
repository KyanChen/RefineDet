import torch
import os
import os.path as op
import json
import torch.backends.cudnn as cudnn
import random
import time
from tensorboardX import SummaryWriter
import cv2
import numpy as np
import matplotlib.pyplot as plt

import Config
import torch.utils.data as data
from utils.SSDDataset import SSDDataset
from utils.Augmentations import SSDAugmentations
from nets.RefineDet_TS import build_refinedet
from nets.layers.RefineMultiBoxLoss import RefineMultiBoxLoss
from nets.layers.PriorBox import PriorBox
from utils.TestNet import test_batch
from utils.ImageTools import get_absolute_bboxes, draw_bboxes, detection_collate, get_img_from_input

if torch.cuda.is_available() and Config.DEVICE == 'gpu':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(Config.MODEL_PATH):
    os.makedirs(Config.MODEL_PATH)

if not os.path.exists(Config.RESULTS_LOG_PATH):
    os.makedirs(Config.RESULTS_LOG_PATH)
if not os.path.exists(op.join(Config.RESULTS_LOG_PATH, 'log')):
    os.makedirs(op.join(Config.RESULTS_LOG_PATH, 'log'))
if not os.path.exists(op.join(Config.RESULTS_LOG_PATH, 'train')):
    os.makedirs(op.join(Config.RESULTS_LOG_PATH, 'train'))
if not os.path.exists(op.join(Config.RESULTS_LOG_PATH, 'test')):
    os.makedirs(op.join(Config.RESULTS_LOG_PATH, 'test'))

if Config.IS_TENSORBOARDX:
    writer = SummaryWriter(op.join(Config.RESULTS_LOG_PATH, 'log'))
net = build_refinedet(Config.INPUT_SIZE, len(Config.CLASSES), is_refine=True)
if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

device = torch.device('cpu')
if torch.cuda.is_available() and Config.DEVICE == 'gpu':
    device = torch.device('cuda')
    net.to(device)
    cudnn.benchmark = True

if Config.IS_TENSORBOARDX:
    net_input_size = torch.zeros(Config.BATCH_SIZE, 3, Config.INPUT_SIZE[0], Config.INPUT_SIZE[1])
    writer.add_graph(net, (net_input_size,))

model_info = {'RESUME_EPOCH': 0, 'RESUME_MODEL': None}
if not op.exists('tools/generate_dep_info/model_info.json'):
    with open('tools/generate_dep_info/model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f)
with open('tools/generate_dep_info/model_info.json', 'r', encoding='utf-8') as f:
    model_info = json.load(f)

if model_info['RESUME_MODEL'] is None or not op.exists(model_info['RESUME_MODEL']):
    model_info['RESUME_EPOCH'] = 0
    print('Loading base network...')

    def xavier(param):
        torch.nn.init.xavier_uniform(param)

    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            m.bias.data.normal_(0, 1 / np.sqrt(n))
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


    base_net_weights = torch.load('nets/model/vgg16_bn.pth', map_location=device)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    # for k, v in net.base_network.named_parameters():

    for k, v in base_net_weights.items():
        if 'features' in k:
            if 'weight' in k:
                name = k[9:]
            elif 'bias' in k:
                name = k[9:]
            else:
                name = k
        else:
            name = k
        new_state_dict[name] = v

    net.base_network.load_state_dict(new_state_dict, strict=False)
    # 将其他两层初始化
    net.base_network[-6:].apply(weight_init)

    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method

    # net.L2Norm_4_3.apply(weight_init)
    # net.L2Norm_5_3.apply(weight_init)
    # for k in net.trans_layers.state_dict():
    #     print(k)
    net.extra_layers.apply(weight_init)
    net.last_layer_trans.apply(weight_init)
    net.trans_layers.apply(weight_init)
    net.up_layers.apply(weight_init)
    net.latent_layers.apply(weight_init)
    net.arm_loc.apply(weight_init)
    net.arm_conf.apply(weight_init)
    net.odm_loc.apply(weight_init)
    net.odm_conf.apply(weight_init)
# load resume network
else:
    print('Loading resume network', model_info['RESUME_MODEL'])
    state_dict = torch.load(model_info['RESUME_MODEL'])
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove module
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)


selected_optimizers = ['SGD', 'Adam', 'RMSprop']
optimizers = [
    torch.optim.SGD(net.parameters(), lr=Config.LR, momentum=Config.MOMENTUM),
    torch.optim.Adam(net.parameters(), lr=Config.LR, betas=(0.9, 0.99)),
    torch.optim.RMSprop(net.parameters(), lr=Config.LR, alpha=0.9)
]
optimizer = optimizers[selected_optimizers.index(Config.OPTIMIZER)]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=30, verbose=True, cooldown=30)

# neg_ratio_to_pos = -1 使用Focal_loss
arm_criterion = RefineMultiBoxLoss(neg_iou_threshold=0.5, pos_iou_threshold=0.5, neg_ratio_to_pos=-1, arm_filter_socre=0.05, is_solve_odm=False)
odm_criterion = RefineMultiBoxLoss(neg_iou_threshold=0.5, pos_iou_threshold=0.5, neg_ratio_to_pos=-1, arm_filter_socre=0.05, is_solve_odm=True)
priors = PriorBox(Config.CFG)()


def train():
    print('Loading Dataset...')
    train_dataset = SSDDataset(pattern='train', transform=SSDAugmentations())
    test_dataset = SSDDataset(pattern='test', transform=SSDAugmentations())
    net.train()
    global model_info
    # loss counters
    current_epoch = model_info['RESUME_EPOCH']
    epoch_size = len(train_dataset) // Config.BATCH_SIZE
    max_iter = Config.MAX_EPOCH * epoch_size
    star_iter = current_epoch * epoch_size

    '''
        dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；
        使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问；
        也可以使用`for inputs, labels in dataloaders`进行可迭代对象的访问；
        shuffle:：是否将数据打乱
        sampler： 样本抽样，后续会详细介绍
        num_workers：使用多进程加载的进程数，0代表不使用多进程
        collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可，default_collate
        pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃 
    '''
    batch_iterator = iter(
        data.DataLoader(train_dataset, Config.BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=True, collate_fn=detection_collate))
    test_batch_iterator = iter(
        data.DataLoader(test_dataset, Config.TEST_BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=True, collate_fn=detection_collate))
    epoch_loss = torch.zeros(2)  # arm_loss, odm_loss
    print("Training...")
    for iteration in range(star_iter, max_iter):

        try:
            img, gt, img_src, img_name, bboxes_src = next(batch_iterator)
        # indicate another epoch
        except StopIteration:

            # write epoch loss
            writer.add_scalars('epoch/train_loss/arm_odm',
                               {'arm_loss': epoch_loss[0], 'odm_loss': epoch_loss[1]}, current_epoch)
            writer.add_scalar('epoch/train_loss/loss', torch.sum(epoch_loss), current_epoch)
            scheduler.step(torch.sum(epoch_loss))
            if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-6:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-6
            epoch_loss = torch.zeros(2)

            if current_epoch % Config.MODEL_SAVE_EPOCH_FREQUENCY == 0:
                model_info = {'RESUME_EPOCH': current_epoch,
                              'RESUME_MODEL': Config.MODEL_PATH + '/model_iter_%d.pth' % iteration}
                torch.save(net.state_dict(), model_info['RESUME_MODEL'])
                with open('tools/generate_dep_info/model_info.json', 'w', encoding='utf-8') as f:
                    json.dump(model_info, f)
            current_epoch += 1

            batch_iterator = iter(
                data.DataLoader(train_dataset, Config.BATCH_SIZE,
                                shuffle=False, num_workers=0, pin_memory=True, collate_fn=detection_collate))
            img, gt, img_src, img_name, bboxes_src = next(batch_iterator)
        img = img.to(device)
        gt = [gt_i.to(device) for gt_i in gt]

        if Config.IS_DEBUG:
            true_bboxes = torch.clone(gt[0])
            true_bboxes = get_absolute_bboxes(true_bboxes, (512, 512), (512, 512))
            img_ = draw_bboxes(
                get_img_from_input(img[0], Config.PRIOR_MEAN_STD['mean'], Config.PRIOR_MEAN_STD['std']), true_bboxes)
            img_path = op.join(
                'Debug', repr(iteration) + '_' + op.basename(img_name[0]).split('.')[0] + Config.IMG_FORMAT)
            cv2.imwrite(img_path, img_)

        if Config.IS_DEBUG:
            print("Train_____________")
            print(gt[0])
            print(img_name[0])

        # forward
        t0 = time.time()
        output = net(img)
        # backward

        optimizer.zero_grad()
        # arm branch loss
        arm_loss_l, arm_loss_c = arm_criterion(output, priors, gt)
        # odm branch loss
        odm_loss_l, odm_loss_c = odm_criterion(output, priors, gt)
        arm_loss = arm_loss_l + arm_loss_c
        odm_loss = odm_loss_l + odm_loss_c
        epoch_loss += torch.as_tensor([arm_loss, odm_loss])
        loss = arm_loss + odm_loss
        if Config.IS_DEBUG:
            print(loss)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if iteration % Config.MODEL_LOG_ITERATION_FREQUENCY == 0:
            writer.add_scalars('iter/train_loss/arm_loc_conf', {'arm_loc': arm_loss_l, 'arm_conf': arm_loss_c}, iteration)
            writer.add_scalars('iter/train_loss/odm_loc_conf', {'odm_loc': odm_loss_l, 'odm_conf': odm_loss_c}, iteration)
            writer.add_scalars('iter/train_loss/arm_odm', {'arm_loss': arm_loss, 'odm_loss': odm_loss}, iteration)
            writer.add_scalar('iter/train_loss/loss_sum', loss, iteration)
            writer.add_scalar('iter/lr', optimizer.state_dict()['param_groups'][0]['lr'], iteration)

        # test model
        if iteration % Config.MODEL_TEST_ITERATION_FREQUENCY == 0 and iteration > Config.MODEL_TEST_ITERATION_FREQUENCY * 1:
            net.eval()
            try:
                img_test, gt_test, img_src_test, img_name_test, bboxes_src_test = next(test_batch_iterator)
            except StopIteration:
                test_batch_iterator = iter(
                    data.DataLoader(test_dataset, Config.TEST_BATCH_SIZE,
                                    shuffle=False, num_workers=0, pin_memory=True, collate_fn=detection_collate))
                img_test, gt_test, img_src_test, img_name_test, bboxes_src_test = next(test_batch_iterator)

            img_test = img_test.to(device)
            gt_test = [gt_test_i.to(device) for gt_test_i in gt_test]

            if Config.IS_DEBUG:
                print("Test_______________")
                print(gt_test[0])
                print(img_name_test[0])
            # forward
            t0_test = time.time()
            output_test = net(img_test)
            predictions_test = test_batch(output_test, priors, iou_threshold=0.5, score_threshold=0.5, is_refine=True)
            t1_test = time.time()
            predictions = test_batch(output, priors, iou_threshold=0.5, score_threshold=0.5, is_refine=True)
            arm_loss_l_test, arm_loss_c_test = arm_criterion(output_test, priors, gt_test)
            arm_loss_test = arm_loss_l_test + arm_loss_c_test
            odm_loss_l_test, odm_loss_c_test = odm_criterion(output_test, priors, gt_test)
            odm_loss_test = odm_loss_l_test + odm_loss_c_test
            loss_test = arm_loss_test + odm_loss_test
            writer.add_scalars('iter/test_loss/arm_odm', {'arm_loss': arm_loss_test, 'odm_loss': odm_loss_test}, iteration)
            writer.add_scalar('iter/test_loss/loss_sum', loss_test, iteration)
            writer.add_scalars('iter/train_test_loss_sum', {'train_loss': loss, 'test_loss': loss_test}, iteration)
            string = 'Iter:%d\tTrain\tloss:%.4f  arm_loss:%.4f  odm_loss:%.4f\tfps:%d\tLR:%.8f\tEpoch:%d'\
                     % (iteration, loss, arm_loss, odm_loss, len(gt) / (t1 - t0), optimizer.state_dict()['param_groups'][0]['lr'], current_epoch)
            string_test = '\t\t\tTest\tloss:%.4f  arm_loss:%.4f  odm_loss:%.4f\tfps:%d'\
                          % (loss_test, arm_loss_test, odm_loss_test, len(gt_test) / (t1_test - t0_test))
            print(string)
            print(string_test)

            # deal with train image
            index = random.sample(range(len(predictions)), k=max(1, int(0.5 * len(predictions))))
            for i in index:
                # return [score, classID, l, t, r, b]
                true_bboxes = get_absolute_bboxes(predictions[i], real_size=Config.INPUT_SIZE)
                img_ = draw_bboxes(
                    get_img_from_input(img[i], Config.PRIOR_MEAN_STD['mean'], Config.PRIOR_MEAN_STD['std']), true_bboxes)
                img_path = op.join(
                    Config.RESULTS_LOG_PATH, 'train',
                    repr(iteration) + '_' + op.basename(img_name[i]).split('.')[0] + '.jpg')
                cv2.imwrite(img_path, img_)
            # deal with test image
            index = random.sample(range(len(predictions_test)), k=max(1, int(0.5 * len(predictions_test))))
            for i in index:
                # return [score, classID, l, t, r, b]
                if Config.IS_SRC_IMG_SIZE_NEAR_NET_SIZE:
                    true_bboxes = get_absolute_bboxes(predictions_test[i], real_size=img_src_test[i].shape[0:2][::-1])
                    img_ = draw_bboxes(img_src_test[i], true_bboxes)
                else:
                    true_bboxes = get_absolute_bboxes(predictions_test[i], real_size=Config.INPUT_SIZE)
                    img_ = draw_bboxes(
                        get_img_from_input(img_test[i], Config.PRIOR_MEAN_STD['mean'], Config.PRIOR_MEAN_STD['std']),
                        true_bboxes)
                img_path = op.join(
                    Config.RESULTS_LOG_PATH, 'test',
                    repr(iteration) + '_' + op.basename(img_name_test[i]).split('.')[0] + '.jpg')
                cv2.imwrite(img_path, img_)
            net.train()


if __name__ == '__main__':
    train()

