import os, glob
from pathlib import Path
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET

def visualize_pascal_voc(img_dir, annot_dir, output_dir, colors = {'damage_crack': (0, 255, 0), 'intrusion_water': (255, 0, 0), 'pipe_connection': (0, 0, 255), 'other': (255, 0, 255)}):
    '''
    PascalVOC可視化
    '''

    rnd_colors = {}

    os.makedirs(output_dir, exist_ok=True)

    for xml_path in glob.glob(os.path.join(annot_dir, '*.xml')):
        print(xml_path)
        xml_file = open(xml_path, encoding='utf-8')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        #対象画像ロード
        img_file = str(Path(xml_path).stem) + '.png'
        img_array = np.fromfile(os.path.join(img_dir, img_file), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if not name.endswith('_detail'):
                # output_dir = r'C:\Users\k_tak\AppData\Local\Temp\annotation_img_detail_output'
                # output_dir = r'C:\Users\k_tak\AppData\Local\Temp\annotation_img_screening_output'
                continue
            xmin = int(float(obj.find('bndbox/xmin').text))
            xmax = int(float(obj.find('bndbox/xmax').text))
            ymin = int(float(obj.find('bndbox/ymin').text))
            ymax = int(float(obj.find('bndbox/ymax').text))

            rnd_color = (0, 0, 0)
            if name not in rnd_colors:
                rnd_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                rnd_colors[name] = rnd_color
            else: 
                rnd_color = rnd_colors[name]

            # ボックス描画
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), rnd_color, thickness=1)
            
            # ラベル
            if (ymax - ymin) > 150:
                img = cv2.rectangle(img, (xmin, ymin), (xmin+170, ymin+15), rnd_color, thickness=-1)
                img = cv2.putText(img, name, (xmin, ymin+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                img = cv2.rectangle(img, (xmin, ymin-10), (xmin+90, ymin), rnd_color, thickness=-1)
                img = cv2.putText(img, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
        ext = '.png'
        _, n = cv2.imencode(ext, img)
        output_img_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '_vis.png')
        with open(output_img_path, mode='w+b') as f:
            n.tofile(f)

        print(output_img_path)

if __name__ == '__main__':
    annot_dir  = r'C:\Users\k_tak\Downloads\annotation_xml'
    img_dir    = r'C:\Users\k_tak\Downloads\230605改善後展開\スクリーニングPoC_損傷判定比較素材'
    output_dir = r'C:\Users\k_tak\AppData\Local\Temp\annotation_img_detail_output'

    colors = {'crack_high': (0, 0, 255), 'crack_low': (255, 178, 255), 'lime_high': (255, 102, 51), 'lime_low': (255, 204, 51), 'peeling': (0 ,217 ,255)}

    visualize_pascal_voc(img_dir, annot_dir, output_dir, colors)
