#-*- coding: euc-kr -*-

import os

def getFileListInPath(path):
  """
  ���� ���� path�� ���� ��θ� return
  :param path:
  :return: file list in path
  """
  file_list = os.listdir(path)
  return file_list

def saveMergedFile(merged_file_name, file_list,path):
  """
  ����Ʈ�� ���� ���� ���� ����Ʈ�� merged_file_name �̸����� ���� �� path�� ����
  :param file_list: ���ϵ� ����
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


file = open('../data/backmyo_novel_1/prerpcessed_bm_novel_utf8_3.txt', 'r', encoding='utf-8')
file2 = open('../data/backmyo_novel_1/prerpcessed_bm_novel_utf8_4.txt', 'w', encoding='utf-8')
while True:
  line= file.readline()

  if not line: break
  tmp_str = line.replace("                   ��             ��             ��",'')
  file2.write(tmp_str)
file.close()
file2.close()









