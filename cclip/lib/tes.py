import torch 



state_dict = torch.load('/remote-home/cs_iot_szy/gaze/lib/ep096.pth')
part_of_model = state_dict['conv21']
print(part_of_model.shape)
