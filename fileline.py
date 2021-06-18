import os
import glob
import pinyin
import shutil

path = r'C:\Users\yeluo\Desktop\SGameServer1'
save_path = r'E:\Anomaly_Prediction-master\dataset\hero\training'

# path = 'C:/Users/Administrator/Desktop/SGameServer2/'
# save_path = 'C:/Users/Administrator/Desktop/hero/testing/'

files = os.listdir(path)
all_count = 0
count_class = 0
for f in files:
	count_hero = 1
	heros_path = os.path.join(path, f)
	heros = os.listdir(heros_path)
	for hero in heros:
		count_class = count_class + all_count
		hero_str = pinyin.get_initial(hero).replace(' ','')
		if len(str(count_hero)) == 1:
			new_name = '0'+str(count_hero)+'_'+hero_str
		else:
			new_name = str(count_hero)+'_'+hero_str
		hero_new_path = os.path.join(save_path,new_name)
		# hero_new_path = os.path.join(save_path, str(count_hero))
		if os.path.exists(hero_new_path):
			pass
		else:
			os.mkdir(hero_new_path)
		hero_use = os.path.join(heros_path, hero)
		path_temp = glob.glob(hero_use+'*/*/1/*/*')
		for i in path_temp:
			hero_new_save_path = os.path.join(hero_new_path, str(count_class)+'.png')
			shutil.copyfile(i, hero_new_save_path)
			count_class += 1

		count_hero += 1
	all_count = len(path_temp)