import tkinter as tk
from tkinter import messagebox

from get_data import half_auto_get_data

# 设置属性名及属性类型
por=[["发票代码","i"],["购买方名称","c"],["发票号码","i"]]

def ui():
    
    # 创建主窗口
    root = tk.Tk()
    root.title("简单的 UI 示例")
    root.geometry("300x400")

    # 标签
    label1 = tk.Label(root, text="请输入读入地址:")
    label1.pack()
    entry1 = tk.Entry(root)
    entry1.pack()
    label2 = tk.Label(root, text="请输入读入地址类型(f:单个文件;d:多个文件):")
    label2.pack()
    entry2 = tk.Entry(root)
    entry2.pack()
    label3 = tk.Label(root, text="请输入输出地址:")
    label3.pack()
    entry3 = tk.Entry(root)
    entry3.pack()

    #    按钮点击事件
    def greet():
        in_path = entry1.get()
        in_type = entry2.get()
        out_path = entry3.get()
        half_auto_get_data(in_path,por,in_type,out_path)
        messagebox.showinfo("完成",f"已经把结果输出到{out_path}")

    # 按钮
    button = tk.Button(root, text="提交", command=greet)
    button.pack(pady=10)

    # 运行主循环
    root.mainloop()