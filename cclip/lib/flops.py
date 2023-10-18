import torch
from utils.utils import get_anchors, get_classes
from thop import profile
from torchstat import stat
from nets.gatector import GaTectorBody
if __name__ == '__main__':
    train_mode=0
    if train_mode==0:
        anchors_mask = [[0, 1, 2]]
    if train_mode==1:
        anchors_mask= [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    class_names, num_classes = get_classes('../data/anchors/voc_classes.txt')
    model = GaTectorBody(anchors_mask, num_classes,train_mode)
    input1 = torch.randn(1, 3, 224, 224)
    input2 = torch.randn(1, 1, 224, 224)
    input3 = torch.randn(1, 3, 224, 224)
    #print(stat(model, (input1,input2,input3,train_mode)))
    flops, params = profile(model, inputs=(input1,input2,input3,train_mode))
    # print("Para gatector:")
    # print("Para:",round(params,5))
    # print("FLOPS:",round(flops,5))
    print("flops: {} params: {} ".format(flops, params))
    print("%.2fG" % (flops/1e9), "%.2fM" % (params/1e6))

