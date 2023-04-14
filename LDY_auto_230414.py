import os
import random




json_root = []
save_root = []
#save_path = "./output_LDY/"


#save_path = "G:\\workspace\\test230413\\"
save_path = "Z:\\PCL\\LDY\\sharing\\"

dir_path = "G:\\3D_Front\\3D-FRONT\\3D-FRONT\\"



future_path = "G:\\3D_Front\\3D-FUTURE-model\\"
texture_path = "G:\\3D_Front\\3D-FRONT-texture\\3D-FRONT-texture\\00cc8b1d-b284-4108-a48f-a18c320a9d3a\\"


count = 0

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.json' in file:


            #file = "4b048449-ce93-4461-9113-2969ee735a95.json"

            file_path = os.path.join(root, file)

            json_root.append(file_path)
            name = os.path.basename(file_path)
            name = name.rstrip(".json")

            #os.mkdir(save_path + name)
            save_root.append(save_path + name)
            
            
            count = count + 1
            if (count==10):
                break
            
        
print("json files : ", len(json_root))
json_root.sort()



#for i in range(0,len(json_root)):
for i in range(0,2000):

    print(i, " : ", len(json_root))

    
    
    if(os.path.isdir(save_root[i])):
        if (i != (len(json_root)-1)):
            if(os.path.isdir(save_root[i+1])):
                continue
    

    os.mkdir(save_root[i])
    cmd = "python -m blenderproc run ./front3d_230414.py " + "--front=" + str(json_root[i]) + " --future_folder=" + future_path + " --front_3D_texture_path=" + texture_path + " --output_dir=" + str(save_root[i])
    print("\n",cmd,"\n")

    
    try:
        os.system(cmd)
    except:
        continue
    

