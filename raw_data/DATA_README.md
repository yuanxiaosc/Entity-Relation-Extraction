
## Download Data
```
print("if not have raw data, please dowload data from http://lic2019.ccf.org.cn/kg !")
def unzip_and_move_files():
    "解压原始文件并且放入 raw_data 文件夹下面"
	#Unzip the original file and put it under the raw_data folder
    os.system("unzip dev_data.json.zip")
    os.system("mv dev_data.json raw_data/dev_data.json")
    os.system("unzip train_data.json.zip")
    os.system("mv train_data.json raw_data/train_data.json")
```

## Data description
This is not the complete data, especially the training data and validation data here are identical, only for code function testing purposes.