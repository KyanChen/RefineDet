import zipfile
z = zipfile.ZipFile("data/datatest/data.zip", "r")
#打印zip文件中的文件列表
for filename in z.namelist( ):
  print(filename)