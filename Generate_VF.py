import argparse
import os
import random
import warnings
from glob import glob

import cv2
import numpy as np
import ray
from tqdm import tqdm
import elasticdeform


@ray.remote
def generate_virtual_flaw(image_path, padding, fade, flaw_type, save_path, sigma, points):
    image_name = image_path.split("/")[-1]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # 이미지를 불러옴.
    image = np.float32(image) # 이미지를 float32로 변환. 결함을 합성할 때 정수가 아니라 실수로 계산하고 나중에 uint8로 변환함.
    origin_image = image.copy() # 원본 이미지 저장. 나중에 비교해서 보여주려고.
    calc_image = image.copy() # 용접부 찾을 이미지.

    # Masking
    PO_IP_mask_image = np.zeros_like(image) 
    Scratch_Leftover_mask_image = np.ones_like(image)
    CT_mask_image = np.zeros_like(image)

    # 용접부 Detection
    calc_image = cv2.rotate(calc_image, cv2.ROTATE_90_CLOCKWISE)
    calc_image = cv2.normalize(calc_image, None, 0, 1000, cv2.NORM_MINMAX, cv2.CV_32F)
    calc_image = cv2.GaussianBlur(calc_image, (31,31), 0)
    calc_image = cv2.resize(calc_image, (512, 512))
    calc_image = np.array(calc_image, dtype=np.float32)
    try:
        for i in range(9):
            calc_image = np.split(calc_image, 2, axis=0)
            calc_image = np.add(calc_image[0], calc_image[1])

        calc_image = np.gradient(np.squeeze(calc_image))
        y1, y2 = 1256 - int(np.argmax(calc_image)* 1256 / 512), 1256 - int(np.argmin(calc_image)* 1256 / 512)
        if y1 < 100 or y2 > 1056:
            raise Exception("y1 or y2 is out of range")
        
        y1 = y1 - padding
        y2 = y2 + padding
        # Masking -> 용접부는 흰색
        PO_IP_mask_image[y1:y2, :] = 1
        # 용접부의 위아래 padding 부분에는 점점 0으로 줄어들게 만들기
        PO_IP_mask_image[y1-fade:y1, :] = np.repeat(np.linspace(0, 1, fade)[:, None], PO_IP_mask_image.shape[1], axis=1)
        PO_IP_mask_image[y2:y2+fade, :] = np.repeat(np.linspace(1, 0, fade)[:, None], PO_IP_mask_image.shape[1], axis=1)
        Scratch_Leftover_mask_image = Scratch_Leftover_mask_image - PO_IP_mask_image
        # Scratch_Leftover_mask_image는 PO_IP_mask_image의 반전
        
        
        CT_mask_image[y1-args.CT_margin:y2+args.CT_margin, :] = 1 # CT는 모재부에도 침범하는 결함을 합성할 수 있도록.
        CT_mask_image[y1-args.CT_margin-fade:y1-args.CT_margin, :] = np.repeat(np.linspace(0, 1, fade)[:, None], CT_mask_image.shape[1], axis=1)
        CT_mask_image[y2+args.CT_margin:y2+args.CT_margin+fade, :] = np.repeat(np.linspace(1, 0, fade)[:, None], CT_mask_image.shape[1], axis=1)

        temp_image = np.zeros_like(image)
        #세로로 이어붙인다. 임시 이미지에 결함을 합성하고 나중에 다시 잘라내고 PO_IP_mask_image를 곱한다.
        temp_image = np.concatenate([temp_image, PO_IP_mask_image, temp_image,], axis=0)
        flaw_image = np.zeros_like(temp_image)
        #float32 
        flaw_image = flaw_image.astype(np.float32)
        y1, y2 = y1 + 1256, y2 + 1256
        
        if flaw_type == "IP":
            try:
                random_flaw_x = np.random.randint(600, 900)
                random_flaw = np.ones((4, random_flaw_x)) * 0.5
                noise = np.random.normal(loc=-0.09, scale=0.3, size=(4, random_flaw_x))
                random_flaw = random_flaw + noise
            
                
                #image의 랜덤한 위치에 IP를 넣는다.
                x = np.random.randint(0, image.shape[1] - random_flaw.shape[1])
                y = int(np.random.normal((y1 + y2) / 2, 20))
                
                flaw_image[y:y+random_flaw.shape[0], x:x+random_flaw.shape[1]] += random_flaw 
                flaw_image = flaw_image[1256:1256*2, :]
                flaw_image = np.multiply(flaw_image, PO_IP_mask_image)
            except Exception as e:
                print(e)
            
        elif flaw_type == "PO":
            random_try = np.random.randint(3, 6)
            for _ in range(random_try):
                random_flaw = np.random.choice(PO_list)
                random_flaw = np.load(random_flaw)
                random_flaw = np.asarray(random_flaw, dtype=np.float32)
                random_flaw = elasticdeform.deform_random_grid(random_flaw, sigma=sigma, points=points, order=1)
                #flaw_image의 랜덤한 위치에 random_flaw를 넣기
                x = np.random.randint(0, image.shape[1] - random_flaw.shape[1])
                y = np.random.randint(y1, y2 - random_flaw.shape[0])
                
                flaw_image[y:y+random_flaw.shape[0], x:x+random_flaw.shape[1]] += random_flaw 
                
            flaw_image = flaw_image[1256:1256*2, :]
            flaw_image = np.multiply(flaw_image, PO_IP_mask_image)
            # 원본이미지와 같은 사이즈의 검정색 이미지 2장을 준비함
            # 1장(flaw_image)은 결함만 두두두두, 1장은 용접부 경계선 스무딩한 것 -> 곱함 -> 용접부 경계선에 있는 결함은 흐려짐
            # 정상이미지에 더함
            
        elif flaw_type == "CT":
            random_flaw = np.random.choice(CT_list)
            random_flaw = np.load(random_flaw)
            random_flaw = np.asarray(random_flaw, dtype=np.float32)
            random_flaw = elasticdeform.deform_random_grid(random_flaw, sigma=sigma, points=points, order=1)
            #flaw_image의 랜덤한 위치에 random_flaw를 넣기
            x = np.random.randint(0, image.shape[1] - random_flaw.shape[1])
            y = np.random.randint(y1-random_flaw.shape[0], y2)
            flaw_image[y:y+random_flaw.shape[0], x:x+random_flaw.shape[1]] += random_flaw * 1
                
            flaw_image = flaw_image[1256:1256*2, :]
            flaw_image = np.multiply(flaw_image, CT_mask_image)
                
        elif flaw_type == "Scratch":
            random_try = np.random.randint(2, 3)
            for _ in range(random_try):
                random_flaw = np.random.choice(Scratch_list)
                random_flaw = np.load(random_flaw)
                random_flaw = np.asarray(random_flaw, dtype=np.float32)
                random_flaw = elasticdeform.deform_random_grid(random_flaw, sigma=sigma, points=points, order=1)
                #flaw_image의 랜덤한 위치에 random_flaw를 넣기
                x = np.random.randint(0, image.shape[1] - random_flaw.shape[1])
                y_top = np.random.randint(y2, y2 + 100)
                y_bottom = np.random.randint(y1 - 100, y1 )
                y = np.random.choice([y_top, y_bottom])
                if y == y_bottom:
                    flaw_image[y-random_flaw.shape[0]:y, x:x+random_flaw.shape[1]] += random_flaw * 1.5
                    
                elif y == y_top:
                    flaw_image[y:y+random_flaw.shape[0], x:x+random_flaw.shape[1]] += random_flaw * 1.5
                
            flaw_image = flaw_image[1256:1256*2, :]
            flaw_image = np.multiply(flaw_image, Scratch_Leftover_mask_image)
            
        elif flaw_type == "Leftover":
            random_try = np.random.randint(3, 4)
            for _ in range(random_try):
                random_flaw = np.random.choice(Leftover_list)
                random_flaw = np.load(random_flaw)
                random_flaw = np.asarray(random_flaw, dtype=np.float32)
                random_flaw = elasticdeform.deform_random_grid(random_flaw, sigma=sigma, points=points, order=1)
                #flaw_image의 랜덤한 위치에 random_flaw를 넣기
                x = np.random.randint(0, image.shape[1] - random_flaw.shape[1])
                y_top = np.random.randint(y2, y2 + 600)
                y_bottom = np.random.randint(y1 - 600, y1 )
                y = np.random.choice([y_top, y_bottom])
                if y == y_bottom:
                    flaw_image[y-random_flaw.shape[0]:y, x:x+random_flaw.shape[1]] += random_flaw * 1
                elif y == y_top:
                    flaw_image[y:y+random_flaw.shape[0], x:x+random_flaw.shape[1]] += random_flaw * 1
            
            flaw_image = flaw_image[1256:1256*2, :]
            flaw_image = np.multiply(flaw_image, Scratch_Leftover_mask_image)

        image += flaw_image
        #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        save_path = save_path + "/" + flaw_type
        os.makedirs(save_path + "/Accept", exist_ok=True)
        os.makedirs(save_path + "/Reject", exist_ok=True)
        os.makedirs(save_path + "/Diff", exist_ok=True)
        
        #합성된 이미지
        cv2.imwrite(save_path + "/Reject/" + image_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #차이 이미지. flaw_image에서 0이 아닌 픽셀은 모두 255로 만들어서 저장
        diff = np.zeros_like(image)
        diff = np.where(flaw_image != 0, 255, 0)
        cv2.imwrite(save_path + "/Diff/" + image_name.replace(".png", "_diff.png"), diff, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #원본이미지
        #origin_image = cv2.normalize(origin_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(save_path + "/Accept/" + image_name, origin_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
    except:
        pass
        
        
        
if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    parser = argparse.ArgumentParser(description='VF Generator')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--flaw_type', type=str)
    parser.add_argument('--padding', type=int, default=-30)
    parser.add_argument('--fade', type=int, default=50)
    parser.add_argument('--nums_worker', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--CT_margin', type=int, default=50)
    parser.add_argument('--sigma', type=int, default=5)
    parser.add_argument('--points', type=int, default=4)

    args = parser.parse_args()
    if args.data_path is None:
        raise Exception("argument 'data_path' must be given.")
    if args.save_path is None:
        raise Exception("argument 'save_path' must be given.")
    if args.flaw_type is None:
        raise Exception("argument 'flaw_type' must be given.")
    
    ray.init(num_cpus=args.nums_worker)
    #seed fix
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    
    image_list = glob(args.data_path + "/*.png")
    print("image_list : ", len(image_list))
    print("flaw_type : ", args.flaw_type)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    PO_list = glob(os.path.join(current_dir, "Extracted_Flaw", "PO", "*.npy"))
    Scratch_list = glob(os.path.join(current_dir, "Extracted_Flaw", "Scratch", "*.npy"))
    Leftover_list = glob(os.path.join(current_dir, "Extracted_Flaw", "Leftover", "*.npy"))
    CT_list = glob(os.path.join(current_dir, "Extracted_Flaw", "CT", "*.npy"))
    
    ray.get([generate_virtual_flaw.remote(image_list[i], padding = args.padding, fade = args.fade, flaw_type = args.flaw_type, save_path = args.save_path, sigma = args.sigma, points = args.points) for i in tqdm(range(len(image_list)))])
