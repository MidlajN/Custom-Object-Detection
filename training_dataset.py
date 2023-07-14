import yaml
import os

def create_yaml():
    data_yaml_path = 'yolov5/data.yaml'
    class_name = input('ENTER THE OBJECT NAME: ')
    data_yaml = {
        'train': '../data/train/images/',
        'val': '../data/val/images/',
        'nc': 1,
        'names': [class_name],
        'train_label': '../data/train/labels/',
        'val_label': '../data/val/labels/',
    }
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
def train_yolov5():
    data_yaml = 'yolov5/data.yaml'
    model_yaml_path = 'yolov5/models/'

    # 'cuda' if torch.cuda.is_available() else 'cpu'

    model_choices = {
        '1': ('yolov5n.yaml', 'yolov5n.pt'),
        '2': ('yolov5s.yaml', 'yolov5s.pt'),
        '3': ('yolov5m.yaml', 'yolov5m.pt'),
        '4': ('yolov5l.yaml', 'yolov5l.pt'),
        '5': ('yolov5x.yaml', 'yolov5x.pt'),
    }

    model_no = input('ENTER THE MODEL NAME : \n 1 . yolov5n \n 2 . yolov5s \n 3 . yolov5m \n 4 . yolov5l \n 5 . yolov5x \n ENTER YOUR CHOICE : ')
    batch = input('ENTER THE BATCH SIZE : ')
    epochs = input('ENTER THE NUMBER OF EPOCHS : ')

    if model_no in model_choices:
        model_yaml, weights = model_choices[model_no]
        model = os.path.join(model_yaml_path, model_yaml)
        command = f'python yolov5/train.py --img 640 --batch {batch} --epochs {epochs} --data {data_yaml} --weights {weights}  --cfg {model} '
        print('EXECUTING THE COMMAND : ', command)
        os.system(command)
        exit()
    else:
        print('INVALID CHOICE')
        exit()
    
if __name__ == '__main__':
    create_yaml()
    train_yolov5()