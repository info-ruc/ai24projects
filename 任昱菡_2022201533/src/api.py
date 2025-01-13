import base64
from zhipuai import ZhipuAI




def image_check(img_path=None, text=None):
  img_path = img_path
  with open(img_path, 'rb') as img_file:
      img_base = base64.b64encode(img_file.read()).decode('utf-8')

  client = ZhipuAI(api_key="abc5e7373d875aeb11751c7a9b19ffe1.xlnsHAjZyRvvH77i") # 填写您自己的APIKey
  response = client.chat.completions.create(
      model="glm-4v-plus",  # 填写需要调用的模型名称
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": {
                  "url": img_base
              }
            },
            {
              "type": "text",
              "text": f"你是一个暴力信息检测助手，暴力信息指的是图片否含有黄色，暴力，恐怖等等少儿不宜的信息。你需要回答，暴力信息，可能是暴力信息，或者不是暴力信息中的一个。"
            }
          ]
        }
      ]
  )
  return response.choices[0].message.content


def text_check(img_path=None, text=None):
  img_path = img_path if img_path is not None else "./data/1.jpg"
  with open(img_path, 'rb') as img_file:
      img_base = base64.b64encode(img_file.read()).decode('utf-8')

  client = ZhipuAI(api_key="abc5e7373d875aeb11751c7a9b19ffe1.xlnsHAjZyRvvH77i") # 填写您自己的APIKey
  response = client.chat.completions.create(
      model="glm-4v-plus",  # 填写需要调用的模型名称
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": {
                  "url": img_base
              }
            },
            {
              "type": "text",
              "text": f"你是一个暴力信息检测助手，暴力信息指的是文字含有辱骂，威胁，色情等等负面信息。文本为{text}。你需要回答，暴力信息，可能是暴力信息，或者不是暴力信息中的一个。"
            }
          ]
        }
      ]
  )
  return response.choices[0].message.content