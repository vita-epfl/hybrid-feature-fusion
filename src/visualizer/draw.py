from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision
import numpy as np

def load_image(path, scale=1.0):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
        image = np.asarray(image) * scale / 255.0
        return image

    
def plot_image_tensor(img_tensor):
    Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
    pil_img = Tensor2PIL(img_tensor)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    plt.imshow(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), aspect='equal')

    
def draw_ped_ann(image, bbox, action):
    thickness = 2
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    if action:
        color = (0, 0, 255)
        org = (int(bbox[0] - 10), int(bbox[1] - 10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .75
        # image = cv2.putText(image, 'walking', org, font, fontScale, color, 2, cv2.LINE_AA)
    else:
        color = (0, 255, 0)
        org = (int(bbox[0]), int(bbox[1] - 10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        # image = cv2.putText(image, 'standing', org, font, fontScale, color, 2, cv2.LINE_AA)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image


class BaseVisualizer:
    def __init__(self, sample):
        self.sample = sample

    def show_frame(self, k=0, title=None):
        """
        Visualize kth frame in history sample
        """
        sample = self.sample
        bbox = sample['bbox'][k]
        action = sample['action'][k]
        s = 'walking' if action else 'standing'
        img_tensor = sample['image'][k]
        pid = sample['id']
        title = f'{pid}, image {k} ({s})' if title is None else title
        Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
        pil_img = Tensor2PIL(img_tensor)
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image = draw_ped_ann(cv2_img, bbox, action)
        plt.title(title)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), aspect='equal')

    def show_history(self, wait_key=0):
        """
        Visualize the whole sequence sample
        """
        sample = self.sample
        pid = sample['id']
        for i in range(sample['image'].size()[0]):
            bbox = sample['bbox'][i]
            action = sample['action'][i]
            img_tensor = sample['image'][i]
            Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
            pil_img = Tensor2PIL(img_tensor)
            opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            image = draw_ped_ann(opencvImage, bbox, action)
            # img = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            # filename = os.path.join(anime_dir, 'img_{:03d}.png'.format(i))
            # cv2.imwrite(filename, img)
            title = f'{pid}, image {i}'
            cv2.imshow(title, image)
            cv2.waitKey(wait_key)
            cv2.destroyAllWindows()

            
