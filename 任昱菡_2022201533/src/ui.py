import streamlit as st
from PIL import Image
import numpy as np
import random
import os
from api import image_check, text_check

# 模拟分类函数
def classify(image, text):
    # 这里只是一个示例，实际应用中需要替换为真实的分类模型
    categories = ['Category 1', 'Category 2', 'Category 3']
    return random.choice(categories)

st.title('暴力信息检测系统')
st.write("注意青少年身心健康")

# 上传图像
image_file = st.file_uploader("请上传图像", type=["jpg", "png", "jpeg"])


if image_file is not None:

    image = Image.open(image_file)
    st.image(image, caption='上传的图像', use_column_width=True)

# 输入文本
text_input = st.text_area("请输入文本")

# 分类按钮
if st.button('分类'):
    if image_file is not None:
        image_path = os.path.join("data", image_file.name)
        print(image_path)
        category = image_check(image_path)
        st.success(f'分类结果：{category}')

    elif text_input:
        # 加载图像
        # image = Image.open(image_file)
        # st.image(image, caption='上传的图像', use_column_width=True)

        # 显示文本
        # st.write('输入的文本：', text_input)

        # 进行分类并显示结果

        print(text_input)
        category = text_check(None, text_input)
        st.success(f'分类结果：{category}')
    else:
        st.error('请上传图像和输入文本后再进行分类。')
