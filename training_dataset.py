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
    model = input('ENTER THE MODEL NAME : \n 1 . yolov5n \n 2 . yolov5s \n 3 . yolov5m \n 4 . yolov5l \n 5 . yolov5x \n ENTER YOUR CHOICE : ')
    batch = input('ENTER THE BATCH SIZE : ')
    epochs = input('ENTER THE NUMBER OF EPOCHS : ')

    # 'cuda' if torch.cuda.is_available() else 'cpu'

    match model:
        case '1':
            model_yaml = os.path.join(model_yaml_path, 'yolov5n.yaml')
            weights = 'yolov5n.pt'
            command = f'python yolov5/train.py --img 640 --batch {batch} --epochs {epochs} --data {data_yaml} --weights {weights}  --cfg {model_yaml} '
            print('EXECUTING THE COMMAND : ', command)
            os.system(command)
            exit()
        case '2':
            model_yaml = os.path.join(model_yaml_path, 'yolov5s.yaml')
            weights = 'yolov5s.pt'
            command = f'python yolov5/train.py --img 640 --batch {batch} --epochs {epochs} --data {data_yaml} --weights {weights}  --cfg {model_yaml} '
            print('EXECUTING THE COMMAND : ', command)
            os.system(command)
            exit()
        case '3':
            model_yaml = os.path.join(model_yaml_path, 'yolov5m.yaml')
            weights = 'yolov5m.pt'
            command = f'python yolov5/train.py --img 640 --batch {batch} --epochs {epochs} --data {data_yaml} --weights {weights}  --cfg {model_yaml} '
            print('EXECUTING THE COMMAND : ', command)
            os.system(command)
            exit()
        case '4':
            model_yaml = os.path.join(model_yaml_path, 'yolov5l.yaml')
            weights = 'yolov5l.pt'
            command = f'python yolov5/train.py --img 640 --batch {batch} --epochs {epochs} --data {data_yaml} --weights {weights}  --cfg {model_yaml} '
            print('EXECUTING THE COMMAND : ', command)
            os.system(command)
            exit()
        case '5':
            model_yaml = os.path.join(model_yaml_path, 'yolov5x.yaml')
            weights = 'yolov5x.pt'
            command = f'python yolov5/train.py --img 640 --batch {batch} --epochs {epochs} --data {data_yaml} --weights {weights}  --cfg {model_yaml} '
            print('EXECUTING THE COMMAND : ', command)
            os.system(command)
            exit()
        case default:
            print('INVALID CHOICE')
            exit()
    
if __name__ == '__main__':
    create_yaml()
    train_yolov5()