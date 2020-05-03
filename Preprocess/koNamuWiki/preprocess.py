# 데이터 경로 /Users/a60058238/Desktop/dev/data/나무위키_덤프_202003.json
# 한국어 나무위키 처리
import json
import ijson

def load_json(filename):
  with open(filename, 'r') as fd:
    parser = ijson.parse(fd)
    for prefix, event, value in parser:
      if prefix.endswith('.title'):
        print("\nTITLE: %s" % value)
      elif prefix.endswith('.text'):
        print("\nCONTENT: %s" % value)

if __name__=='__main__':
  file_path = '/Users/a60058238/Desktop/dev/data/나무위키_덤프_202003.json'
  load_json(file_path)



