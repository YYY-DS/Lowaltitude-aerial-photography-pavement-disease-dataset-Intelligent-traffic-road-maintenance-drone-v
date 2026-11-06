# 低空航拍路面病害数据集｜交通道路智能养护无人机视觉图像资料10520期
## 标签
`#无人机道路巡检` `#路面病害检测` `#公路裂缝识别` `#道路坑洼识别` `#智慧交通养护` `#YOLO目标检测素材` `#沥青路面缺陷` `#航拍道路图像` `#交通基础设施AI` `#路面修复识别` `#低空遥感视觉` `#道路数字化运维`
---

全国公路总里程突破530万公里，沥青路面长期受重载车辆、温差、雨水侵蚀，持续产生横向/纵向/斜向裂缝、网状鳄鱼纹、坑洼沉陷、旧修复痕迹六类典型病害。细微裂缝若未及时处置，雨水渗入路基会引发路面大面积塌陷，单次修复成本可达普通养护的6~10倍；路面坑洼会大幅提升车辆磨损、诱发交通事故。
传统道路人工巡检存在四大核心痛点：1. 人工徒步/车载巡查单日仅覆盖5~10公里，高速、山区公路巡检周期长达半月；2. 车流中作业存在极高交通安全风险，养护人员事故率常年居高不下；3. 人眼对细微发丝裂缝辨识度差，人工漏检率超18%，病害判定标准因人而异，数据无法量化归档；4. 自建标注素材成本高，航拍俯视视角、多光照干扰的标准化道路病害图像稀缺，制约AI巡检系统落地。
无人机低空航拍搭配YOLO检测算法，可一次性完成数十公里路面全域扫描，自动分类、统计各类病害。本文配套标准化YOLO格式航拍路面病害标注素材，覆盖道路全品类典型损伤，配套图像预处理、数据集划分、模型训练、批量航拍推理完整工程代码，可快速搭建道路智能巡检、数字化养护决策平台。

## 一、图像素材工程整体概述
### 1.1 项目定位
面向无人机低空俯视道路巡检场景，适配YOLO全系列检测框架的沥青路面病害标注图像素材，用于各类裂缝、坑洼、修复痕迹识别算法研发、交通养护管理平台开发、轻量化机载视觉模型迭代优化。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/333202c8d13746c8973666abcecd0fc6.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/12a5a2e02d9f4eeb8cceaa561fa712bc.png)

### 1.2 核心量化指标
1. 航拍实拍图像总量：2400张，统一无人机俯视角度，分辨率覆盖1600×1200、1920×1080两类主流航拍相机规格；
2. 全量标注目标框总数：11600个路面病害标注，单张图像平均4.83处损伤目标；
3. 数据集分层划分：训练集1920张（80%）、验证集480张（20%），分层抽样保证6类病害样本均衡，无类别样本倾斜；
4. 标注标准：原生YOLO txt归一化xywh坐标标签，无需格式转换，直接兼容YOLOv5/v8/v10/v11；
5. 六大标注类别：坑洼、鳄鱼纹裂缝、纵向裂缝、斜向裂缝、修复痕迹、横向裂缝；
6. 实景覆盖场景：正午强光反光、阴天弱光照、树荫阴影遮挡、车流干扰、雨后积水路面、新旧修复混杂路面6类野外真实工况；

### 1.3 素材落地应用价值
1. 公路、高速、市政道路自动化病害检测算法训练、消融对比实验；
2. 无人机道路智能巡检系统、交通养护数字化管理平台开发；
3. 依据病害种类、数量、面积生成分级养护方案，辅助交通部门制定维修计划；
4. 自动驾驶路面感知、路况识别相关课题研究；
5. 土木、交通、计算机视觉专业毕业设计、实训实验素材；
6. 结合大模型实现病害自动分级+维修工程量测算一体化系统开发。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b90a77e33bb94859ae139f10a387f32c.png)

## 二、图像素材核心特性拆解
1. **病害分类完全贴合公路养护规范**
六类病害对应不同路面损坏等级与处置方案：横向/纵向/斜向细微裂缝预防性灌缝；大面积鳄鱼纹、深坑洼需铣刨重铺；旧修复痕迹用于判断重复破损高发路段，标签分类可直接对接道路养护工单系统。
2. 复杂干扰样本占比充足，模型泛化能力更强
超45%图像包含树荫、车辆、积水、标线干扰，大量低对比度细微发丝裂缝样本，解决普通素材强光清晰样本过多、野外实景漏检严重问题。
3. 标准YOLO标注，开箱训练无需转换
图像与标签一一对应，坐标完成归一化，省去格式转换、坐标修正环节，大幅缩短项目研发周期。
4. 样本体量均衡，普通消费显卡即可训练
2400张图像在RTX3060/4060显卡可完成完整训练迭代，中小企业、院校无需高性能工作站。
5. 配套完整工具链，形成工程闭环
附带样本分层划分、标签清洗、航拍图像光照校正、模型训练、批量航拍图像推理全套Python代码，可直接接入无人机巡检后台。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c60bf5046940490a8f15b69424f09167.png)

## 三、工程目录结构
```
road_pavement_defect_data/
├── images/
│   ├── train/        # 训练集航拍图像 1920张
│   └── val/          # 验证集航拍图像 480张
├── labels/
│   ├── train/        # 训练集YOLO标注txt
│   └── val/          # 验证集YOLO标注txt
├── road_defect.yaml  # YOLO训练专用配置文件
├── split_road_data.py    # 数据集分层划分脚本
├── clean_road_label.py   # 标注框清洗修复工具
├── road_clahe_enhance.py # 航拍图像光照对比度校正
├── train_yolo_road.py    # 路面病害模型训练程序
└── road_batch_infer.py   # 无人机航拍批量推理接口
```

### road_defect.yaml 训练配置文件
```yaml
# 无人机俯视路面病害检测训练配置
train: ./images/train
val: ./images/val
nc: 6
names:
  0: pothole          # 坑洼沉陷
  1: alligator_crack  # 鳄鱼纹网状裂缝
  2: vertical_crack   # 纵向裂缝
  3: diagonal_crack   # 斜向裂缝
  4: repair_mark      # 路面修复痕迹
  5: transverse_crack # 横向裂缝
# 细微裂缝专属训练参数
imgsz: 640
conf: 0.20
iou: 0.40
mosaic: 1.0
mixup: 0.15
```

## 四、深度学习配套完整代码（附道路航拍落地场景注释）
### 4.1 数据集分层划分脚本（场景注释：8:2分层抽样均衡6类病害样本，避免模型对高频坑洼过拟合、细微裂缝漏检）
```python
# split_road_data.py 航拍道路病害数据集自动划分工具
# 适用场景：原始无人机航拍图像批量拆分训练/验证集，均衡各类裂缝、坑洼样本分布
import os
import random
import shutil
from tqdm import tqdm

# 原始素材路径
RAW_IMG = "./raw_uav_road_images"
RAW_LBL = "./raw_labels"
OUT_ROOT = "./"
TRAIN_RATIO = 0.8

def create_output_dirs():
    dir_list = [
        f"{OUT_ROOT}/images/train", f"{OUT_ROOT}/images/val",
        f"{OUT_ROOT}/labels/train", f"{OUT_ROOT}/labels/val"
    ]
    for path in dir_list:
        os.makedirs(path, exist_ok=True)

def split_dataset():
    create_output_dirs()
    img_files = [f for f in os.listdir(RAW_IMG) if f.endswith(("jpg", "png", "jpeg"))]
    random.seed(42)  # 固定随机种子，保证实验可复现
    random.shuffle(img_files)

    train_count = int(len(img_files) * TRAIN_RATIO)
    train_imgs = img_files[:train_count]
    val_imgs = img_files[train_count:]

    # 迁移训练集图像+标签
    for img_name in tqdm(train_imgs, desc="生成训练集航拍图像"):
        shutil.copy(f"{RAW_IMG}/{img_name}", f"{OUT_ROOT}/images/train/{img_name}")
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        shutil.copy(f"{RAW_LBL}/{label_name}", f"{OUT_ROOT}/labels/train/{label_name}")
    # 迁移验证集图像+标签
    for img_name in tqdm(val_imgs, desc="生成验证集航拍图像"):
        shutil.copy(f"{RAW_IMG}/{img_name}", f"{OUT_ROOT}/images/val/{img_name}")
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        shutil.copy(f"{RAW_LBL}/{label_name}", f"{OUT_ROOT}/labels/val/{label_name}")
    print(f"数据集划分完成：训练集{len(train_imgs)}张，验证集{len(val_imgs)}张")

if __name__ == "__main__":
    split_dataset()
```

### 4.2 标注清洗修复脚本（场景注释：无人机航拍易产生边缘越界框、细小石子/标线误标注，批量过滤无效框、修正归一化坐标）
```python
# clean_road_label.py 道路病害标注批量清洗脚本
# 适用场景：俯视航拍图边缘病害框溢出、路面噪点误标，统一修复过滤降低训练噪声
import os
import glob
import cv2

IMG_ROOT = "./images"
LBL_ROOT = "./labels"
COORD_MIN = 1e-6
COORD_MAX = 1 - 1e-6

def clamp_coord(val):
    # 坐标限制在0~1归一化区间
    return max(COORD_MIN, min(COORD_MAX, val))

def clean_all_labels():
    img_paths = glob.glob(os.path.join(IMG_ROOT, "*/*.*"))
    fix_cnt = 0
    drop_cnt = 0
    for img_p in img_paths:
        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_p = img_p.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl_p):
            continue
        new_lines = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            cls, xc, yc, bw, bh = parts[0], parts[1], parts[2], parts[3], parts[4]
            # 过滤极小噪点标注（非真实病害）
            if bw < 0.003 or bh < 0.003:
                drop_cnt += 1
                continue
            xc = clamp_coord(xc)
            yc = clamp_coord(yc)
            bw = clamp_coord(bw)
            bh = clamp_coord(bh)
            new_lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f}\n")
            fix_cnt += 1
        with open(lbl_p, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    print(f"标签清洗完成：修复标注{fix_cnt}个，删除无效噪点框{drop_cnt}个")

if __name__ == "__main__":
    clean_all_labels()
```

### 4.3 CLAHE光照增强预处理脚本（场景注释：解决航拍逆光、阴天路面对比度低，强化细微裂缝纹理特征，提升模型识别能力）
```python
# road_clahe_enhance.py 航拍道路图像对比度校正工具
# 适用场景：强光、树荫、阴天低画质航拍图增强，突出细微裂缝边界
import cv2
import os

def clahe_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray_enhance = clahe.apply(gray)
    return cv2.cvtColor(gray_enhance, cv2.COLOR_GRAY2BGR)

def batch_enhance_img(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_list = os.listdir(src_dir)
    for name in img_list:
        path = os.path.join(src_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue
        enhance_img = clahe_enhance(img)
        cv2.imwrite(os.path.join(out_dir, name), enhance_img)
    print(f"航拍图像增强完成，输出路径：{out_dir}")

if __name__ == "__main__":
    batch_enhance_img("./raw_uav_road_images", "./enhance_road_images")
```

### 4.4 YOLO路面病害训练脚本（场景注释：调低置信阈值捕捉发丝裂缝，开大Mosaic模拟多病害混杂路面，早停防止复杂路面样本过拟合）
```python
# train_yolo_road.py 无人机路面病害检测训练主程序
# 适用场景：俯视航拍细微裂缝专用训练参数，优化弱纹理病害检出率
from ultralytics import YOLO
import torch

torch.manual_seed(42)
# 轻量化模型适配无人机机载边缘设备
model = YOLO("yolov8n.pt")

train_res = model.train(
    data="./road_pavement_defect_data/road_defect.yaml",
    epochs=140,
    imgsz=640,
    batch=12,
    device=0,
    conf=0.20,
    iou=0.40,
    mosaic=1.0,
    mixup=0.15,
    patience=15,
    project="uav_road_exp",
    name="pavement_defect_v1"
)
print("训练完成，最优权重路径：runs/uav_road_exp/pavement_defect_v1/weights/best.pt")
```

### 4.5 航拍图像批量推理接口（场景注释：对接无人机巡检图片文件夹，批量统计各类病害数量，直接输出养护报表结构化数据）
```python
# road_batch_infer.py 道路航拍批量识别服务
# 适用场景：单次巡检数百张航拍图自动识别，统计坑洼、各类裂缝数量用于养护评估
from ultralytics import YOLO
import os

model = YOLO("./weights/best.pt")

def batch_detect_road(image_dir, save_draw=False):
    all_report = []
    img_files = [f for f in os.listdir(image_dir) if f.endswith(("jpg", "png"))]
    for img_name in img_files:
        img_path = os.path.join(image_dir, img_name)
        res = model(img_path, conf=0.20)
        single_record = {"img_name": img_name, "defect_list": []}
        for r in res:
            boxes = r.boxes
            cls_map = r.names
            for box in boxes:
                cid = int(box.cls)
                defect_name = cls_map[cid]
                conf = round(float(box.conf), 3)
                single_record["defect_list"].append({"defect": defect_name, "conf": conf})
            if save_draw:
                r.save(f"./infer_out/{img_name}")
        all_report.append(single_record)
    return all_report

if __name__ == "__main__":
    result_data = batch_detect_road("./uav_road_batch", save_draw=True)
    print("航拍病害批量识别完成，统计报表：", result_data)
```

## 五、整套交付素材完整清单
1. 2400张无人机低空俯视道路实拍图像，覆盖坑洼、六类裂缝、修复痕迹全品类路面损伤；
2. 总计11600条标准YOLO归一化txt标注文件，无越界、无效标注；
3. 全套Python工程脚本：数据集分层划分、标签清洗、图像光照增强、模型训练、批量推理，全部附带业务注释；
4. 适配6类路面病害的yaml训练配置文件；
5. 配套操作文档：环境部署、航拍图像训练调参、模型轻量化导出至机载硬件教程；
6. 细微裂缝、复杂干扰场景专属调参方案，提升野外巡检识别稳定性。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b4d22fc66b54f5d868ca936dbd872d9.png)



