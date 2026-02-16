"""
Dataset Loader for Basel's VIA-annotated Car Damage Dataset
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CarDamageDataset(Dataset):
    """
    Loads car damage dataset with VIA polygon annotations
    """
    
    def __init__(self, root_dir, subset='train', transforms=None):
        """
        Args:
            root_dir: Path to dataset (D:\hero\Automated-Car-Damage-Detection\dataset)
            subset: 'train' or 'val'
            transforms: Albumentations or torchvision transforms
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transforms = transforms
        
        # Path to subset directory
        self.subset_dir = os.path.join(root_dir, subset)
        self.images_dir = self.subset_dir
        
        # Load VIA annotations
        annotation_file = os.path.join(self.subset_dir, 'via_region_data.json')
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter out images without annotations and get image list
        self.image_data = []
        for key, value in self.annotations.items():
            if value.get('regions') and len(value['regions']) > 0:
                filename = value['filename']
                image_path = os.path.join(self.images_dir, filename)
                if os.path.exists(image_path):
                    self.image_data.append({
                        'filename': filename,
                        'path': image_path,
                        'regions': value['regions']
                    })
        
        print(f"✅ Loaded {len(self.image_data)} annotated images from {subset} set")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        # Load image
        img_data = self.image_data[idx]
        img = Image.open(img_data['path']).convert('RGB')
        img_width, img_height = img.size
        
        # Extract polygon annotations
        regions = img_data['regions']
        
        masks = []
        boxes = []
        
        for region_id, region in regions.items():
            shape_attrs = region.get('shape_attributes', {})
            
            if shape_attrs.get('name') == 'polygon':
                # Get polygon points
                all_x = shape_attrs.get('all_points_x', [])
                all_y = shape_attrs.get('all_points_y', [])
                
                if len(all_x) < 3 or len(all_y) < 3:
                    continue
                
                # Create binary mask from polygon
                from skimage.draw import polygon
                rr, cc = polygon(all_y, all_x, shape=(img_height, img_width))
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[rr, cc] = 1
                
                # Get bounding box from mask
                pos = np.where(mask)
                if len(pos[0]) == 0:
                    continue
                
                ymin, ymax = pos[0].min(), pos[0].max()
                xmin, xmax = pos[1].min(), pos[1].max()
                
                # Skip invalid boxes
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                boxes.append([xmin, ymin, xmax, ymax])
                masks.append(mask)
        
        # Convert to tensors
        if len(boxes) == 0:
            # No valid annotations - return empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, img_height, img_width), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # All class 1 (damage)
            masks_tensor = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        # Convert image to tensor
        img_tensor = T.ToTensor()(img)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks_tensor,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,)),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.int64)
        }
        
        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)
        
        return img_tensor, target
    
    def get_image_info(self, idx):
        """Get image metadata"""
        img_data = self.image_data[idx]
        return {
            'filename': img_data['filename'],
            'path': img_data['path'],
            'num_regions': len(img_data['regions'])
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))
