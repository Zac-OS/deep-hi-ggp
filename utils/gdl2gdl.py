import sys
import re
text = open(sys.argv[1]).read()
text = re.sub(r'([A-Z])', r'?\1', text).lower()
for i in range(20):
    text = re.sub(r'(:- .+)&', r'\1\n ', text)

text = re.sub(r'\(([^,\(\)]+),', r' (\1),', text)
text = re.sub( r',([^,\(\)]+),', r' (\1),', text)
text = re.sub( r',([^,\)\)]+)\)', r' (\1)', text)
# text = re.sub(r'\(([^( ]+)\)', r' \1 ', text)

text = text.replace('&', '')
text = re.sub(r':-( .+)', r'\n \1', text)
text = text.replace(':-', '')
text = re.sub(r'~(.*\(.+)', r'not (\1)', text)
text = re.sub(r'~(.+)', r'not \1', text)
text = re.sub(r'(  .+)\n([^ ])', r'\1)\n\2', text)
text = re.sub(r'\n  (.+)', r'\n    (\1)', text)
text = re.sub(r'\n([^ \n].+)\n    \(', r'\n  (<= (\1)\n    (', text)
text = re.sub(r'\n([^ \n;].+)', r'\n  (\1)', text)

text = re.sub(r'[ ]+\)', ')', text)
text = re.sub(r'\([ ]+', '(', text)
text = re.sub(r'\(([^( ]+)\)', r' \1 ', text)
text = re.sub(r'([^\n ])[ ]+', r'\1 ', text)




with open(sys.argv[2], "w+") as f:
    # f.write("\n".join(text))
    f.write(text)
    f.write(")")
