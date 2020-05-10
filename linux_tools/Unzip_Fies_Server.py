import zipfile

# zip_src: 需要解压的文件路径
# dst_dir: 解压后文件存放路径
def unzip_file(zip_src, dst_dir):
	r = zipfile.is_zipfile(zip_src)
	if r:
		fz = zipfile.ZipFile(zip_src, 'r')
		for file in fz.namelist():
			fz.extract(file, dst_dir)
	else:
		print('This is not a zip file !!!')

if __name__ == '__main__':
    zip_src = '/home/aistudio/data/data30872/WKK.zip'
    dst_dir = '/home/aistudio/mydata/WKK'
    unzip_file(zip_src, dst_dir)
    
    # zip_src = '/home/aistudio/RefineDet.zip'
    # dst_dir = '/home/aistudio'
    # unzip_file(zip_src, dst_dir)
    
    # zip_src = '/home/aistudio/data/data29650/vgg16.zip'
    # dst_dir = '/home/aistudio/nets/model'
    # unzip_file(zip_src, dst_dir)