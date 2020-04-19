#-*- coding: euc-kr -*-

import os

def getFileListInPath(path):
  """
  전달 받은 path의 파일 경로를 return
  :param path:
  :return: file list in path
  """
  file_list = os.listdir(path)
  return file_list

def saveMergedFile(merged_file_name, file_list,path):
  """
  리스트로 전달 받은 파일 리스트를 merged_file_name 이름으로 병합 후 path에 저장
  :param file_list: 파일들 리스
  :return:
  example:
    path = "../data/backmyo_novel_1/"
    file_list = os.listdir(path)

    saveMergedFile('merged_bm_novel_utf8.txt',file_list, path)
  """
  save_file = open(path+merged_file_name, 'w', encoding='utf-8')
  for file_name in file_list:
    merge_file = open(path+file_name, 'r',encoding='euc-kr')
    while True:
      try:
        line = merge_file.readline()
      except Exception as ex:
        print(file_name + str(ex))
      save_file.write(line)
      if not line: break
    merge_file.close()
  save_file.close()
  return

def txt2euc_kr(txt_path, tran_txt_parh):
  txt_path = open(txt_path, 'r',encoding='euc-kr')
  tran_txt_parh = open(tran_txt_parh, 'w', encoding='utf-8')

  while True:
    try:
      line = txt_path.readline()
      tran_txt_parh.write(line)
    except Exception as ex:
      print(ex)
    # save_file.write(line)
    if not line: break

  txt_path.close()
  tran_txt_parh.close()

  return


def removeFirstSpace():
  file = open('../data/backmyo_novel_1/merged_bm_novel_utf8.txt', 'r', encoding='utf-8')
  file2 = open('../data/backmyo_novel_1/prerpcessed_bm_novel_utf8_.txt', 'w', encoding='utf-8')
  while True:
    try:
      line = file.readline()
      if line[0]==' ':
        line= line[1:]
      file2.write(line)
    except Exception as ex:
      print(ex)
    # save_file.write(line)
    if not line: break
  file.close()
  file2.close()
  return

if __name__ == "__main__":
  txt2euc_kr('/Users/a60058238/Desktop/dev/workspace/nlp/Data/fairy_tale.txt', '/Users/a60058238/Desktop/dev/workspace/nlp/Data/fairy_tale_utf-8.txt')








