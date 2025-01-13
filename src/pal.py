from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import pandas as pd

# 初始化 OCR 模型
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 支持中文

# 输入图片路径
image_path = "ocr\\sharpened_image.jpg"
# 文字检测和识别
results = ocr.ocr(image_path, cls=True)

data = pd.DataFrame(columns=["公司","姓名","收款人","票据号","时间","金额"])
# 输出识别结果
for line in results[0]:
    print(f"识别文字: {line[1][0]}, 置信度: {line[1][1]}")


# 可视化结果
image = Image.open(image_path).convert('RGB')
boxes = [line[0] for line in results[0]]
txts = [line[1][0] for line in results[0]]
scores = [line[1][1] for line in results[0]]

# 绘制结果
from paddleocr import draw_ocr
import matplotlib.pyplot as plt

result_image = draw_ocr(image, boxes, txts, scores)
plt.imshow(result_image)
plt.show()

data=pd.DataFrame({"company":"company1"})
data.add("skd")

