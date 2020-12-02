import re
import os

# os.path.basename(os.path.dirname(os.path.realpath(__file__)))
# realPath = os.path.realpath(currentFile)
# print(p)
cur_path = os.path.dirname(os.path.abspath(__file__)) 

def process(text): 
    text = re.sub(r'\n', r'', text)
    text = re.sub(r'^\s+', r'', text)
    text = re.sub(r'\t', r',', text)
    text = re.sub(r'\\\[Theta\]', r'th', text)
    text = re.sub(r'\[t\]', r'', text)
    text = re.sub(r'Sin\[([a-z]+)\]', r's\1', text)
    text = re.sub(r'Cos\[([a-z]+)\]', r'c\1', text)
    text = re.sub(r'Sin\[q\+th\]', r'sqth', text)
    text = re.sub(r'Sin\[q\+2 th\]', r'sq2th', text)
    text = re.sub(r'Cos\[q\+th\]', r'cqth', text)
    text = re.sub(r'Cos\[q\+2 th\]', r'cq2th', text)
    text = re.sub(r'\(([a-z]+)\^\\\[Prime\]\)', r'd\1', text)
    text = re.sub(r'\^', r'**', text)
    text = re.sub(r' ', r'*', text)
    return text


with open(cur_path + "/mathematica_format.txt") as fp: 
    Lines = fp.readlines() 


for i in range(len(Lines)):
    if Lines[i] == '\n':
        Lines[i] = '], dtype=np.float)\n\n' + 'h = np.array([\n' 
        i += 1
        break
    else:
        Lines[i] = ' '*4 + '['+ process(Lines[i]) + '],\n'

for j in range(i, len(Lines)):
    if Lines[j] == '\n':
        Lines[j] = '], dtype=np.float)\n\n' + 'peef = np.array([\n' 
        j += 1
        break
    else:
        Lines[j] = ' '*4+ process(Lines[j]) + ',\n'

for k in range(j, len(Lines)):
    if Lines[k] == '\n':
        Lines[k] = '], dtype=np.float)\n\n' + 'Jeef = np.array([\n' 
        k += 1
        break
    else:
        Lines[k] = ' '*4+ process(Lines[k]) + ',\n'

for m in range(k, len(Lines)):
    Lines[m] = ' '*4 + '['+ process(Lines[m]) + '],\n'
        
Lines[0] = 'M = np.array([\n' + Lines[0]
Lines[-1] = Lines[-1] + '], dtype=np.float)\n'

with open(cur_path + "/python_format.txt", 'w') as fp:
    for ele in Lines:
        fp.write(ele)