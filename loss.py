import torch
from torch import nn
from torch.nn import functional as F
from customize_dataset import CustomCocoForYolo
from torch.utils.data import DataLoader




class Loss(nn.Module):
    def __init__(self, grid_sz=7, num_bboxes=2, num_classes=3, lambda_coord=0.5, lambda_noobj=0.5):
        super().__init__()
        
        self.S = grid_sz
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def IoU_calcute(self, box1, box2):
        # box1 and box2 are in form of [x1, y1, x2, y2]

        pass


    def forward(self, pred_tens_batch, target_tens_batch):
        """
        Args:
            pred_tens_batch (Tensor), target_tens_batch (Tensor)
            both target_tens_batch and pred_tens_batch are in shape of [batch_size, S, S, 5*B+C] the 5 is [x, y, w, h, conf]
        Return:
            loss, a tensor of [1]
        
        """
        S, B, C = self.S, self.B, self.C
        N = B * 5 + C

        obj_mask = target_tens_batch[:, :, :, 4] > 0     # mask for the cells which contain objects. [n_batch, S, S]
        obj_mask = obj_mask.unsqueeze(-1).expand_as(target_tens_batch)     # [n_batch, S, S] -> [n_batch, S, S, N]
        
        noobj_mask = target_tens_batch[:, :, :, 4] == 0  # mask for the cells which do not contain objects. [n_batch, S, S]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tens_batch) # [n_batch, S, S] -> [n_batch, S, S, N]


        ### start: calculate loss for cells containing no objects (noobj)
        noobj_target = target_tens_batch[noobj_mask].view(-1, N)
        noobj_pred = pred_tens_batch[noobj_mask].view(-1, N)

        noobj_conf_element_mask = torch.zeros(noobj_target.shape, dtype=torch.bool)
        for b in range(B):
            noobj_conf_element_mask[:, b*5+4] = 1
        
        noobj_target_conf_values = noobj_target[noobj_conf_element_mask]
        noobj_pred_conf_values = noobj_pred[noobj_conf_element_mask]

        loss_conf_noobj = F.mse_loss(noobj_pred_conf_values, noobj_target_conf_values, reduction='sum')
        ### end: calculate loss for cells containing no objects (noobj)


        ### start: calculate loss for cells containing objects (obj)
        obj_target = target_tens_batch[obj_mask].view(-1, N)        # get targets in the form of vectors, regardless of other dims. [n_obj, N] (n_obj: is the num of cells that include object)
        bbox_target = obj_target[:, :5*B].contiguous().view(-1, 5)  # [n_obj x B, 5] (5 is [x, y, w, h, conf])
        class_target = obj_target[:, 5*B:]                          # [n_obj, C]

        obj_pred = pred_tens_batch[obj_mask].view(-1, N)        # get targets in the form of vectors, regardless of other dims. [n_obj, N] (n_obj: is the num of cells that include object)
        bbox_pred = obj_pred[:, :5*B].contiguous().view(-1, 5)  # [n_obj x B, 5] (5 is [x, y, w, h, conf])
        class_pred = obj_pred[:, 5*B:]                          # [n_obj, C]





        ### end: calculate loss for cells containing objects (obj)




        loss_xy = F.mse_loss()
        loss_wh = F.mse_loss()
        loss_conf_obj = F.mse_loss()
        loss_class = F.mse_loss()

        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_conf_obj + self.lambda_noobj * loss_conf_noobj + loss_class

        return loss
    




if __name__ == '__main__':
    val_data = CustomCocoForYolo('./data/images/val/', 
                      './data/annotation/instances_val2017.json', 
                      img_sz=448, grid_sz=5, num_boxes=2, num_classes=3, classes_name=['human', 'car', 'dog'])

    train_loader = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=1)

    criterion = Loss(grid_sz=5, num_bboxes=2, num_classes=3, lambda_coord=0.5, lambda_noobj=0.5)
    for i, (imgs, targets) in enumerate(train_loader):
        # loss = criterion(None, targets)
        loss = criterion(targets, targets)
