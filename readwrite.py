import io
def read(file_path,tagged=True):
    output=[]
    with io.open(file_path,'r',encoding='utf-8') as file:
        for line in file:
            line=line.strip()
            output.append([])
            words=line.split(' ')
            for word in words:
                if tagged:
                    splitted=word.split('/')
                    output[-1].append([word[0:len(word)-len(splitted[-1])-1],splitted[-1]])
                else:
                    output[-1].append(word)
    return output

tag=lambda x:x[0]+'/'+x[1]

def writeOutput(tagging,file_path='hmmoutput.txt'):
    with io.open(file_path,'w',encoding='utf-8') as file:
        lines=[]
        for line in tagging:
            output=' '.join([tag(word) for word in line])
            output+='\n'
            lines.append(str(output))
        file.writelines(lines)