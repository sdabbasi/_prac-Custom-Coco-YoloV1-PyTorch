import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import json, math, os


class CustomCocoForYolo(Dataset):
    def __init__(self, img_folder_path, annot_file_path, img_sz=448, grid_sz=7, num_boxes=2, num_classes=3, classes_name=['person', 'car', 'dog']):
        self.img_folder_path, self.annot_file_path = img_folder_path, annot_file_path
        self.S = grid_sz          # called S in the paper
        self.B = num_boxes        # called B in the paper
        self.C = num_classes      # called C in the paper
        self.classes_name = classes_name
        self.transforms = self.get_transformes(img_sz)
        self.images, self.target_tensors = self.process_data(img_folder_path, annot_file_path)
        


    def process_data(self, img_folder_path, annot_file_path):
        with open(annot_file_path) as f:
            annot_json = json.load(f)
        categ_ids, list_valid_annot, distinct_images = self.select_annotations(annot_json)
        images, grouped_boxes, grouped_classes = self.gather_boxes_labels(categ_ids, list_valid_annot, distinct_images)
        target_tensors = self.prepare_target_tensor(images, grouped_boxes, grouped_classes)

        return images, target_tensors
        

    # collect images and annotations related to input classes
    def select_annotations(self, annot_json):
        categ_ids = []
        for category in annot_json['categories']:
            if category['name'] in self.classes_name:
                categ_ids.append(category['id'])

        list_valid_annot = list(filter(lambda annot:annot['category_id'] in categ_ids and annot['iscrowd'] == 0, annot_json['annotations']))
        
        distinct_images_id = set()
        for valid_annot in list_valid_annot:
            distinct_images_id.add(valid_annot['image_id'])
        distinct_images = list(filter(lambda img:img['id'] in distinct_images_id, annot_json['images']))

        return categ_ids, list_valid_annot, distinct_images


    # gather all boxes and labels related to each image together
    def gather_boxes_labels(self, categ_ids, list_valid_annot, distinct_images):
        ordered_grouped_boxes, ordered_grouped_classes = [], []
        for img in distinct_images:
            selected = list(filter(lambda annot:annot['image_id']==img['id'], list_valid_annot))
            boxes, classes = [], []
            for annot in selected:
                boxes.append(annot['bbox'])     #each bbox is as (x1, y1, width, height)
                classes.append(categ_ids.index(annot['category_id']))
            ordered_grouped_boxes.append(boxes)
            ordered_grouped_classes.append(classes)
        
        return distinct_images, ordered_grouped_boxes, ordered_grouped_classes


    # prepare S * S * (B * 5 + C) target tensor; the "5" value is (x, y, w, h, confidence)
    def prepare_target_tensor(self, images, grouped_boxes, grouped_classes):
        S, B, C = self.S, self.B, self.C
        cell_size = 1.0/S       # this is normal cell_size; it will be used when finding the cell in grid to which the object belongs
        all_prepared_targets = []

        for ind, img in enumerate(images):
            normalizing_vector = torch.tensor([img['width'], img['height'], img['width'], img['height']])
            boxes = torch.tensor(grouped_boxes[ind], dtype=torch.int32)
            classes = grouped_classes[ind]
            targ = torch.zeros(S, S, B * 5 + C)

            for indx, box in enumerate(boxes):
                label = classes[indx]
                x, y, w, h = box/normalizing_vector   # first normalize the values regarding the image's width and height, box is as (x, y, w, h)
                x_cent, y_cent = x+w/2, y+h/2
                grid_idx_x_coor = math.floor(x_cent/cell_size)   # find the grid of box; index on x coordinate
                grid_idx_y_coor = math.floor(y_cent/cell_size)   # find the grid of box; index on y coordinate
                norm_x_cent = (x_cent/cell_size)-grid_idx_x_coor      # x and y related to each cell have to get normalized regarding their grids 
                norm_y_cent = (y_cent/cell_size)-grid_idx_y_coor      # x and y related to each cell have to get normalized regarding their grids 
                
                for b in range(B):
                    targ[grid_idx_y_coor, grid_idx_x_coor, b*5:(b+1)*5] = torch.tensor([norm_x_cent, norm_y_cent, w, h, 1.0])
                targ[grid_idx_y_coor, grid_idx_x_coor, B*5+label] = 1.0

            all_prepared_targets.append(targ)

        return all_prepared_targets


    def get_transformes(self, img_sz):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_sz, img_sz))
        ])
       

    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, indx):
        img = Image.open(os.path.join(self.img_folder_path, self.images[indx]['file_name']))
        img = self.transforms(img)
        return img, self.target_tensors[indx]


    def draw_and_save(self, indx):
        img = Image.open(os.path.join(self.img_folder_path, self.images[indx]['file_name']))
        target = self.target_tensors[indx]

        img_h = self.images[indx]['height']
        img_w = self.images[indx]['width']

        for j in range(ins.S):
            for i in range(ins.S):
                if target[j, i, 4] == 1.0:
                    x_cent = int((i+target[j, i, 0])*(img_w/ins.S))
                    y_cent = int((j+target[j, i, 1])*(img_h/ins.S))
                    w = int(target[j, i, 2] * img_w)
                    h = int(target[j, i, 3] * img_h)

                    img_draw = ImageDraw.Draw(img)
                    shape = [(x_cent-2, y_cent-2), (x_cent+2, y_cent+2)]
                    img_draw.rectangle(shape, fill=None, outline ="green", width=2)
                    shape = [(x_cent-int(w/2), y_cent-int(h/2)), (x_cent+int(w/2), y_cent+int(h/2))]
                    img_draw.rectangle(shape, fill=None, outline ="green", width=2)
        
        img.save(f'''./data/images/visualize_test/{self.images[indx]['file_name']}.jpg''', format='JPEG')



if __name__ == '__main__':
    ins = CustomCocoForYolo('./data/images/val', './data/annotation/instances_val2017.json')

    for i in range(10):
        ins.draw_and_save(i)

