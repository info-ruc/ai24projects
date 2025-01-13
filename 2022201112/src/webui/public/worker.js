// worker.js
let typeWriterInProgress = false;
let messageQueue = [];
let isQueueProcessing = false; // 用于标记队列是否在处理中
let timeoutId = null; // 用于设置超时检测

// 启动队列处理
function processQueue() {
  if (messageQueue.length === 0) {
    if (timeoutId) {
      clearTimeout(timeoutId);  // 如果队列为空且在等待超时清理，清理它
      timeoutId = null;
    }
    return; // 队列为空时，停止处理
  }

  // 如果当前有消息在打印，则不继续
  if (typeWriterInProgress) {
    // 如果当前有任务正在进行，继续检查队列
    return;
  }

  // 处理队列中的第一条消息
  const nextMessage = messageQueue.shift();
  typeWriterInProgress = true; // 标记打字机效果开始

  // 通知主线程打印消息
  postMessage({ action: 'printMessage', message: nextMessage });

  // 设置一个定时器，若在一段时间内队列没有新消息，则认为打印完毕
  timeoutId = setTimeout(() => {
    if (messageQueue.length === 0 && !typeWriterInProgress) {
      postMessage({ action: 'finish' });  // 所有消息打印完毕
    }
  }, 1000);  // 1秒内队列空且没有消息在打印则结束处理
}

// 监听主线程发来的消息
onmessage = function (e) {
  const { action, data } = e.data;

  switch (action) {
    case 'enqueue':
      messageQueue.push(data);  // 将新消息加入队列
      break;
    case 'finishWriting':
      typeWriterInProgress = false;  // 标记当前消息打印完成
      break;
    default:
      break;
  }
};

// 定时检查队列（循环）
setInterval(() => {
  processQueue();  // 每次检查队列是否有待处理的消息
}, 50);  // 每50ms检查一次队列

