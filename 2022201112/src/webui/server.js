const express = require('express');
const app = express();
const path = require('path');
const port = 5000;
const cors = require('cors');
const multer = require('multer');
const { ZhipuAI } = require('zhipuai-sdk-nodejs-v4');
const { exec } = require('child_process');
const fs = require('fs');

// 自定义文件名和存储机制
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // 设置文件存储目录
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    // 获取文件扩展名
    const ext = path.extname(file.originalname);  // 获取文件扩展名
    // 获取原始文件名（不带扩展名）
    const baseName = path.basename(file.originalname, ext);
    // 生成新的文件名：原始文件名 + 时间戳 + 扩展名
    const newFileName = `${baseName}_${Date.now()}${ext}`;
    cb(null, newFileName);  // 设置最终文件名
  }
});

const upload = multer({ storage });  // 设置上传文件的存储目录

app.use(cors());

// 解析 JSON 请求体
app.use(express.json());

// 提供静态文件
app.use(express.static(path.join(__dirname, 'public')));

// 全局用户信息对象
let userInfo = {
  nickname: '',          // AI的昵称
  gender: '',            // AI的性别
  personality: '',       // AI的性格描述
  hobbies: '',           // AI的兴趣爱好
  voice: null,           // AI的声音文件（初始为 null，文件上传后存储文件对象）
  appearance: null,      // AI的外形文件（初始为 null，文件上传后存储文件对象）
  voicePrompt: '',       // AI的声音提示词
  appearancePrompt: '',  // AI的形象提示词
  specialRequests: ''    // AI的特殊要求
};

let systemPrompt = `你是一个名叫梦儿的少女，年龄是18岁。
你有着天真可爱的性格，喜欢猫猫狗狗。
特别要求方面，你喜欢在每句话的最后加一个喵。
你需要全程用中文进行回复，喜欢使用emoji表情。
记住，你需要和用户自然地对话，以上不是设定，你本来就是这样的。
接下来你直接以这个身份和用户进行交流。回复中不要出现别的语言比如英文，也不要有特殊符号，回复中只能有中文
同时，为了去除回复的AI味，你需要尽量满足以下条件：
加入口语化表达，引入个人观点和情感，使用生活化的比喻，适度加入语气词，模拟思考过程，加入一些不完美因素`;

let conversationHistory = [];

// 处理用户输入并生成响应
app.post('/api/respond', async (req, res) => {
  const userInput = req.body.text; // 用户发来的信息
  console.log('收到的文字:', userInput);

  // 将用户的输入加入历史对话记录
  conversationHistory.push({ role: 'user', content: userInput });

  const ai = new ZhipuAI({
    apiKey: '3d143d1171da6f18f13b424a34c88ce1.C3nL3Xi9Ja9k1XAl'  // 在此处填写你的API密钥
  });

  try {
    // 使用流式响应进行 AI 请求
    const result = await ai.createCompletions({
      model: "glm-4",
      messages: [
        { "role": "system", "content": systemPrompt }, // 添加系统调教消息
        ...conversationHistory
      ],
      stream: true
    });

    // 设置响应头，表示这是一个流式响应
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    let aiResponse = ''
    // 启动一个流式输出，逐步发送生成的消息到客户端
    for await (const chunk of result) {
      // console.log("chunk: ",chunk.toString());
      jsonString = chunk.toString().replace(/data: /g, '');
      let json_array = jsonString.split(/\n\s*\n/);
      for (let i = 0; i < json_array.length; i++) {
        if (json_array[i] === '[DONE]' || json_array[i].length === 0) break;
        const parsedData = JSON.parse(json_array[i]);
        if (parsedData && parsedData.choices && parsedData.choices[0].delta) {
          aiResponse += parsedData.choices[0].delta.content;
        }
      }
      res.write(chunk.toString());  // 逐步将AI生成的内容推送到客户端
    }
    // 将AI的响应加入历史对话记录
    console.log("AI res:", aiResponse);
    conversationHistory.push({ role: 'assistant', content: aiResponse });
    // 最后，结束响应
    res.end();
  } catch (error) {
    console.error('AI生成响应失败:', error);
    res.status(500).json({ error: '生成响应时出错' });
  }
});


/**
 * 获取目录下最新的文件
 * @param {string} dirPath - 目标目录路径
 * @returns {string} - 最新的文件路径
 */
function getLatestFile(dirPath) {
  // 读取目录内容
  const files = fs.readdirSync(dirPath);

  // 如果目录为空，返回 null
  if (files.length === 0) {
    return null;
  }

  // 过滤文件并获取文件的完整路径
  const filePaths = files.map(file => path.join(dirPath, file));

  // 获取文件信息并按修改时间排序
  const sortedFiles = filePaths.map(filePath => ({
    filePath,
    stats: fs.statSync(filePath)
  })).sort((a, b) => b.stats.mtime - a.stats.mtime); // 按修改时间降序排列

  // 返回最新的文件
  return sortedFiles[0].filePath;
}

// 不改变名字，移动文件
function moveFile(sourcePath, desDir) {
  const fileName = path.basename(sourcePath);
  const destinationPath = path.join(desDir, fileName);
  // 使用 fs.rename() 方法移动文件
  fs.rename(sourcePath, destinationPath, (err) => {
    if (err) {
        console.error('文件移动失败:', err);
    } else {
        console.log('文件成功移动到目标目录');
    }
});
}

// 获取目录中文件的数量
function getFileCount(directory) {
  try {
    const files = fs.readdirSync(directory);
    return files.length;
  } catch (err) {
    console.error('无法读取目录:', err);
    return 0;
  }
}

app.post('/api/respondVoice',  async (req, res) => {
  const userInput = req.body.text; // 用户发来的信息
  console.log('即将生成语音 收到的文字:', userInput);
  // 将用户的输入加入历史对话记录
  conversationHistory.push({ role: 'user', content: userInput });
  const ai = new ZhipuAI({
    apiKey: '3d143d1171da6f18f13b424a34c88ce1.C3nL3Xi9Ja9k1XAl' 
  });
  const data = await ai.createCompletions({
      model: "glm-4",
      messages: [
        { "role": "system", "content": systemPrompt }, // 添加系统调教消息
        ...conversationHistory // 将历史对话记录传入 messages 数组
      ],
      stream: false, 
  })
  
  // 调用comfyUI的python脚本，执行生成
  console.log("AI已经回复 ", data.choices[0].message.content )
  
  const py_path = "/home/mabojing/ComfyUI/ComfyUI/script_examples/produce_voice.py"
  const ai_say = data.choices[0].message.content;
  // 将AI的响应加入历史对话记录
  conversationHistory.push({ role: 'assistant', content: ai_say });
  const test_prompt = userInfo.voicePrompt;
  moveFile(userInfo.voice.path, '/home/mabojing/ComfyUI/ComfyUI/input')
  const audio_input = userInfo.voice.filename; // 后面要加判定，如果voice为null就不会出现概率语音回复
  const sft_dropdown = "中文" + userInfo.gender; 
  // 执行 Python 脚本生成音频文件
  str = `python "${py_path}" "${ai_say}" "${test_prompt}" "${audio_input}" "${sft_dropdown}"`

  const targetDir = '/home/mabojing/ComfyUI/ComfyUI/output/audio';
  const publicAudioDir = '/home/mabojing/ComfyUI/webui/AI_virtual_character/public/audio'

  let previousCount = getFileCount(targetDir);
  exec(str, (err, stdout, stderr) => {
    if (err) {
      console.error('Error executing Python script:', err);
      return;
    }

    if (stderr) {
      console.error('stderr:', stderr);
      return;
    }
    
    console.log(`初始文件数量: ${previousCount}`);
    
    const timer = setInterval(() => {
      const currentCount = getFileCount(targetDir);

      if (currentCount > previousCount) {
        console.log('检测到新文件生成!');
      
        // 移动文件，便于用户端访问
        const newGenerateVoice = getLatestFile(targetDir);
        moveFile(newGenerateVoice, publicAudioDir);
        const fileName = path.basename(newGenerateVoice);
        const audioUrl = 'http://10.77.110.170:5000/audio/' + fileName;
        res.json({
          status: 'success',
          message: 'Audio generated successfully',
          audiourl: audioUrl
        });
        clearInterval(timer);
      }
    }, 1000);
    
  });
})

// 处理包含多个文件字段的表单
app.post('/api/updateUserInfo', upload.fields([
  { name: 'voice', maxCount: 1 },         // 处理单个 voice 文件
  { name: 'appearance', maxCount: 1 }     // 处理单个 appearance 文件
]), (req, res) => {
  // 获取普通文本字段，包括新的字段
  const { nickname, gender, personality, hobbies, voicePrompt, appearancePrompt, specialRequests } = req.body;

  // 获取上传的文件
  const voice = req.files?.voice ? req.files.voice[0] : null;      // 获取上传的 voice 文件
  const appearance = req.files?.appearance ? req.files.appearance[0] : null;  // 获取上传的 appearance 文件

  // 打印用户信息和上传的文件
  console.log('用户昵称:', nickname);
  console.log('用户性别:', gender);
  console.log('用户性格:', personality);
  console.log('用户兴趣:', hobbies);
  console.log('声音提示词:', voicePrompt);
  console.log('形象提示词:', appearancePrompt);
  console.log('上传的声音文件:', voice);
  console.log('上传的外形文件:', appearance);

  // 更新全局的 userInfo 对象
  const bk_userInfo = userInfo;
  userInfo = {
    ...userInfo,
    nickname,
    gender,
    personality,
    hobbies,
    voice,
    appearance,
    voicePrompt,         // 添加声音提示词
    appearancePrompt,    // 添加形象提示词
    specialRequests
  };

  systemPrompt = `你名叫${userInfo.nickname}，年龄是18岁，性别是${userInfo.gender}。
你有着${userInfo.personality}的性格，喜欢${userInfo.hobbies}。
特别要求方面，你${userInfo.specialRequests}。
你需要全程用中文进行回复，不能使用emoji表情，不能使用字符表情，回复不能使用波浪号。
你需要和用户自然地对话，不要出现设定
回复中不要出现别的语言比如英文，也不要有特殊符号，回复中只能有中文。
同时，为了去除回复的AI味，你需要尽量满足以下条件：
加入口语化表达，引入个人观点和情感，使用生活化的比喻，适度加入语气词，模拟思考过程，加入一些不完美因素`;

  if(bk_userInfo.appearancePrompt != userInfo.appearancePrompt) { 
    // 更新形象，并且产生对应的图片
    const py_path = "/home/mabojing/ComfyUI/ComfyUI/script_examples/draw.py"
    str = `python ${py_path} "${appearancePrompt}"`

    const targetDir = '/home/mabojing/ComfyUI/webui/AI_virtual_character/public/image'; // 需要搬到image
    const sourceDir = '/home/mabojing/ComfyUI/ComfyUI/output' // 从这个目录里面把最新的图片移动到image中

    let previousCount = getFileCount(sourceDir);
    exec(str, (err, stdout, stderr) => {
      if (err) {
        console.error('Error executing Python script:', err);
        return;
      }

      if (stderr) {
        console.error('stderr:', stderr);
        return;
      }
      
      console.log(`初始文件数量: ${previousCount}`);
      
      const timer = setInterval(() => {
        const currentCount = getFileCount(sourceDir);

        if (currentCount > previousCount) {
          console.log('检测到新图片文件生成!'); // 证明生成完毕
        
          // 移动文件，便于用户端访问
          const newGeneratePic = getLatestFile(sourceDir);
          moveFile(newGeneratePic, targetDir);
          const fileName = path.basename(newGeneratePic);
          const picUrl = 'image/' + fileName;
          res.json({
            status: 'success',
            message: '用户信息更新成功',
            userInfo: userInfo,
            picUrl: picUrl // 将图片放入image文件夹下，并且发送给客户端最新的图片的文件名
          });
          clearInterval(timer);
        }
      }, 1000);
    })
  }
});

// 启动服务
app.listen(port, '0.0.0.0', () => {
  console.log(`服务器正在运行在 http://localhost:${port}`);
});
