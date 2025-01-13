const chatContainer = document.getElementById('chatContainer');
const inputText = document.getElementById('inputText');
const sendButton = document.getElementById('sendButton');

// 创建一个全局变量来存储当前的消息框
let currentMessageDiv = null;
// 存储用户头像和服务器头像
let userAvatar = 'image/user.jpg'
let serverAvatar = 'image/server.jpg'
let canVoice = false;

// 获取当前时间字符串
function getCurrentTime() {
  const now = new Date();
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  return `${hours}:${minutes}`;
}

// 添加消息到聊天窗口
function addMessage(content='', type, isAudio = false, audioUrl = '') {
  console.log("enter");
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('message', type);

  const avatarImg = document.createElement('img');
  avatarImg.classList.add('avatar');
  avatarImg.src = type === 'user' ? userAvatar : serverAvatar; // 替换为实际头像路径

  const contentDiv = document.createElement('div');
  contentDiv.classList.add('content');
  if (isAudio) {
    const audioElement = document.createElement('audio');
    audioElement.controls = true;
    audioElement.src = audioUrl + '?t=' + new Date().getTime(); // 设置音频链接
    audioElement.classList.add('audio-player');
    contentDiv.appendChild(audioElement);
  } else {
    contentDiv.textContent = content;
  }

  const timestampDiv = document.createElement('div');
  timestampDiv.classList.add('timestamp');
  timestampDiv.textContent = getCurrentTime();

  messageDiv.appendChild(avatarImg);
  messageDiv.appendChild(contentDiv);
  messageDiv.appendChild(timestampDiv);

  chatContainer.appendChild(messageDiv);

  // 自动滚动到底部
  chatContainer.scrollTop = chatContainer.scrollHeight;
}


// 创建 Web Worker 实例
const worker = new Worker('worker.js');

// 监听 Worker 返回的消息
worker.onmessage = function (e) {
  const { action, message } = e.data;

  if (action === 'printMessage') {
    if (!currentMessageDiv) {
      // 如果当前消息框尚未创建，则创建一个新的
      currentMessageDiv = createMessageDiv('server');
    }
    typeWriter(message, currentMessageDiv).then(() => {
      worker.postMessage({ action: 'finishWriting' });  // 通知 Worker 打印完成
    });
  } else if (action === 'finish') {
    console.log('所有消息已打印完毕');
  }
};

function getVoiceResponse() {
  const probability = 0.5;  // 50%的概率请求语音回复
  return Math.random() < probability;
}

// 发送消息
sendButton.addEventListener('click', () => {
  const userInput = inputText.value.trim();
  if (!userInput) return;
  // 显示用户消息
  addMessage(userInput, 'user');
  inputText.value = '';
  const isVoice = getVoiceResponse();
  
  if(!canVoice || !isVoice) {
    // 发送请求到服务端
    fetch('http://10.77.110.170:5000/api/respond', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: userInput}),
    })
      .then((response) => {
        currentMessageDiv = null;
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        // 逐步读取流中的数据
        reader.read().then(function processText({ done, value }) {
          if (done) return;

          // 解码流数据
          let now_read = decoder.decode(value, { stream: true });
          now_read = now_read.replace(/data: /g, '');
          let json_array = now_read.split(/\n\s*\n/);

          try {
            for (let i = 0; i < json_array.length; i++) {
              if (json_array[i] === '[DONE]' || json_array[i].length === 0) break;
              const parsedData = JSON.parse(json_array[i]);
              if (parsedData && parsedData.choices && parsedData.choices[0].delta) {
                const aiContent = parsedData.choices[0].delta.content;
                if (aiContent) {
                  // 将新消息加入到队列
                  worker.postMessage({ action: 'enqueue', data: aiContent });
                }
              }
            }
          } catch (error) {
            console.error('JSON解析错误:', error);
          }

          // 继续读取数据流
          reader.read().then(processText);
        });
      })
      .catch((error) => {
        console.error('请求失败:', error);
        addMessage('请求失败，请稍后再试。', 'server');
      });
  } else {
    addMessage('说话中...','server');
    // 发送请求到服务端
    fetch('http://10.77.110.170:5000/api/respondVoice', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: userInput })
    }).then(response => response.json())
    .then(data => {
      if (data.status === 'success') {
        const audiourl = data.audiourl;
        console.log("Get response: ", audiourl);
        // 处理收到的音频文件的url
        addMessage('','server',true,audiourl);
      } else {
          console.error('Error:', data.message);
      }
    }).catch(error => {
      console.error('Request failed', error);
    });
  }
});

// 打字机效果
function typeWriter(text, messageDiv) {
  return new Promise((resolve) => {
    const contentDiv = messageDiv.querySelector('.content');
    let i = 0;
    const interval = setInterval(() => {
      contentDiv.textContent += text[i];
      i++;
      if (i === text.length) {
        clearInterval(interval);
        resolve(); // 打字机效果结束，调用 resolve
      }
    }, 50); // 每50毫秒显示一个字符
  });
}

// 创建消息框
function createMessageDiv(type) {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('message', type);

  const avatarImg = document.createElement('img');
  avatarImg.classList.add('avatar');
  avatarImg.src = type === 'user' ? userAvatar : serverAvatar; // 替换为实际头像路径

  const contentDiv = document.createElement('div');
  contentDiv.classList.add('content');
  messageDiv.appendChild(avatarImg);
  messageDiv.appendChild(contentDiv);

  const timestampDiv = document.createElement('div');
  timestampDiv.classList.add('timestamp');
  timestampDiv.textContent = getCurrentTime();
  messageDiv.appendChild(timestampDiv);

  chatContainer.appendChild(messageDiv);

  // 自动滚动到底部
  chatContainer.scrollTop = chatContainer.scrollHeight;

  return messageDiv;
}

// 按 Enter 键发送消息
inputText.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendButton.click();
  }
});


// // 定义背景图片数组
// const backgrounds = [
//   'image/background2.jpg',
//   'image/background3.webp'
// ];

// 当前背景图片索引
let currentBackgroundIndex = 0;

// 函数：切换背景，并且也切换头像
function changeBackground(pic_url) {
  // 设置新的背景图片，并使用 cover 保证背景图片自适应容器
  const pic_url_with_timestamp = `${pic_url}?t=${new Date().getTime()}`;
  chatContainer.style.backgroundImage = `url(${pic_url_with_timestamp})`;
  serverAvatar = pic_url_with_timestamp; // 换头像
  chatContainer.style.backgroundSize = 'cover';         // 使图片覆盖整个容器
  chatContainer.style.backgroundPosition = 'center';    // 背景居中显示
  chatContainer.style.backgroundRepeat = 'no-repeat';   // 防止背景图重复
  
  // // 循环背景图片索引
  // currentBackgroundIndex = (currentBackgroundIndex + 1) % backgrounds.length;
}

// // 设置每隔 5 秒切换一次背景
// setInterval(changeBackground, 5000);

// // 初始背景设置
// changeBackground();





// 获取侧边栏、按钮和主内容区域
const sidebar = document.getElementById('sidebar');
const toggleSidebarButton = document.getElementById('toggleSidebarButton');
const main = document.querySelector('main');
const inputContainer = document.querySelector('.input-container');
const header = document.querySelector('header');

// 点击按钮时切换侧边栏显示/隐藏
toggleSidebarButton.addEventListener('click', () => {
  // 切换侧边栏的显示状态
  sidebar.classList.toggle('open'); // 切换 'open' 类，控制侧边栏滑动
  main.classList.toggle('sidebar-open'); // 切换 'sidebar-open' 类，调整主内容区域
  
  // 切换按钮的状态
  toggleSidebarButton.classList.toggle('open'); // 切换按钮状态
  
  // 控制聊天框和输入框的交互状态
  if (sidebar.classList.contains('open')) {
    header.classList.add('blocked');
    chatContainer.classList.add('blocked');  // 当侧边栏打开时禁用聊天框
    inputContainer.classList.add('blocked'); // 当侧边栏打开时禁用输入框
  } else {
    header.classList.remove('blocked');
    chatContainer.classList.remove('blocked');  // 当侧边栏关闭时恢复聊天框
    inputContainer.classList.remove('blocked'); // 当侧边栏关闭时恢复输入框
  }
});



document.getElementById('submitButton').addEventListener('click', function () {
  // 获取表单中的输入值
  const nickname = document.getElementById('nickname').value;
  const gender = document.querySelector('input[name="gender"]:checked')?.value;
  const personality = document.getElementById('personality').value;
  const hobbies = document.getElementById('hobbies').value;
  const voice = document.getElementById('voice').files[0];
  if (voice != null) {
    canVoice = true;
  } // 如果提交了一次音频的话，那么就允许语音对话了
  const appearance = document.getElementById('appearance').files[0];
  const voicePrompt = document.getElementById('voicePrompt').value;  // 获取声音提示词
  const appearancePrompt = document.getElementById('appearancePrompt').value;  // 获取形象提示词
  const specialRequests = document.getElementById('specialRequests').value;

  // 显示加载中的提示
  const loadingOverlay = document.createElement('div');
  loadingOverlay.id = 'loadingOverlay';
  loadingOverlay.style.position = 'fixed';
  loadingOverlay.style.top = '0';
  loadingOverlay.style.left = '0';
  loadingOverlay.style.width = '100%';
  loadingOverlay.style.height = '100%';
  loadingOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  loadingOverlay.style.display = 'flex';
  loadingOverlay.style.justifyContent = 'center';
  loadingOverlay.style.alignItems = 'center';
  loadingOverlay.style.zIndex = '9999';
  loadingOverlay.innerHTML = '<div style="color: white; font-size: 20px;">虚拟形象生成中...</div>';
  document.body.appendChild(loadingOverlay);

  // 禁止点击其他区域
  document.body.style.pointerEvents = 'none';

  // 使用 FormData 发送数据（包括文件）
  const formData = new FormData();
  formData.append('nickname', nickname);
  formData.append('gender', gender);
  formData.append('personality', personality);
  formData.append('hobbies', hobbies);
  formData.append('voice', voice);
  formData.append('appearance', appearance);
  formData.append('voicePrompt', voicePrompt);  // 添加声音提示词
  formData.append('appearancePrompt', appearancePrompt);  // 添加形象提示词
  formData.append('specialRequests', specialRequests);

  // 发送数据到服务器
  fetch('/api/updateUserInfo', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    console.log('Server response:', data.userInfo);
    // 修改背景
    changeBackground(data.picUrl);
    alert('修改成功！');
  })
  .catch(error => {
    console.error('Error:', error);
    alert('提交失败');
  })
  .finally(() => {
    // 请求完成后，隐藏加载提示并恢复页面操作
    document.body.removeChild(loadingOverlay);
    document.body.style.pointerEvents = 'auto';
  });
});






