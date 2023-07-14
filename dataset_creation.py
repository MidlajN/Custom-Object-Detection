import cv2
import os, shutil
import glob, random


class DatasetCreation():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.yolov5_path = 'yolov5'
        if not os.path.exists(self.yolov5_path + '/train.py'):
            shutil.rmtree(self.yolov5_path)
            print('---------- YOU HAVE NOT CLONED THE YOLOv5 REPOSITORY ---------- \nEXECUTING COMMAND : ')
            os.system('git clone https://github.com/ultralytics/yolov5')
        self.train_dir = 'yolov5/data/train/'
        self.valid_dir = 'yolov5/data/val/'
        self.tracker = cv2.TrackerMIL_create()

    def remove_files(self, dir):
        files = glob.glob(os.path.join(dir, '*'))
        [os.remove(file) for file in files]
    
    def dataset_dir(self):
        train_img_dir = os.path.join(self.train_dir, 'images')
        valid_img_dir = os.path.join(self.valid_dir, 'images')
        train_label_dir = os.path.join(self.train_dir, 'labels')
        valid_label_dir = os.path.join(self.valid_dir, 'labels')

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(valid_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(valid_label_dir, exist_ok=True)

        self.remove_files(train_img_dir)
        self.remove_files(valid_img_dir)
        self.remove_files(train_label_dir)
        self.remove_files(valid_label_dir)
        
        return (
            train_img_dir,
            valid_img_dir,
            train_label_dir,
            valid_label_dir
        )


    def split_dataset(self):
        training_img_dir, validation_img_dir, training_label_dir, validation_label_dir = self.dataset_dir()
        image_dir = os.path.join('dataset/images')
        label_dir = os.path.join('dataset/labels')
        split_ratio = 0.8

        images = os.listdir(image_dir)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        training_images = images[:split_index]
        validation_images = images[split_index:]

        training_labels = [os.path.splitext(image)[0] + '.txt' for image in training_images]
        validation_labels = [os.path.splitext(image)[0] + '.txt' for image in validation_images]

        self.move_files(training_images, training_labels, image_dir, label_dir, training_img_dir, training_label_dir)
        self.move_files(validation_images, validation_labels, image_dir, label_dir, validation_img_dir, validation_label_dir)
        print('Training Images :', len(training_images))
        print('Validation Images :', len(validation_images))


    def move_files(self, images, labels, src_img_dir, src_label_dir, training_img_dir, training_label_dir):
        for image, label in zip(images, labels):
            src_image_path = os.path.join(src_img_dir, image)
            dst_image_path = os.path.join(training_img_dir, image)
            shutil.move(src_image_path, dst_image_path)

            src_label_path = os.path.join(src_label_dir, label)
            dst_label_path = os.path.join(training_label_dir, label)
            shutil.move(src_label_path, dst_label_path)

    
    def create_dataset(self):
        ret, frame = self.cap.read()
        bbox = cv2.selectROI(frame, False)
        self.tracker.init(frame, bbox)
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break   
            ok, bbox = self.tracker.update(frame)
            if ok:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                height, width, _ = frame.shape
                norm_x = (x + w / 2) / width
                norm_y = (y + h / 2) / height
                norm_w = w / width
                norm_h = h / height 

                cv2.imwrite(f'dataset/images/frame_{frame_count}.jpg', frame)
                with open(f'dataset/labels/frame_{frame_count}.txt', 'w') as f:
                    f.write(f'0 {norm_x} {norm_y} {norm_w} {norm_h}')
                print('Saved frame_{}.jpg to : /dataset'.format(frame_count))
                frame_count += 1

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()

obj = DatasetCreation()
obj.create_dataset()
obj.split_dataset()
cv2.destroyAllWindows()
         
