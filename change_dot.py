import os
os.system('pyreverse -o dot code/')
encode = "utf-8"
fontname = "TimesNewRoman" # 黑体:SimHei 新宋体:SimSun 仿宋:NSimSun 宋体:FangSong
content = ''
with open('classes.dot', 'br') as f:
    content = f.read()
# 更改字体
content = content.replace("shape=\"record\"".encode(encode),("shape=\"record\", fontname="+fontname).encode(encode))
# 去除多余空的区域
content = content.replace("\\l|".encode(encode),"".encode(encode))
with open('classes_note.dot', 'bw') as f:
    f.write(content)
os.system('dot -Tpdf classes_note.dot -o classes_note.pdf')