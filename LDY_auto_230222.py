import os
import random


'''
number = []

# 무한 반복문 실행
while True:
    random_number = random.randint(1, 100000)

    # 랜덤숫자가 리스트에 없을 경우 리스트에 추가
    if random_number not in number :
        number.append(random_number)
    
    # 리스트의 길이가 20이면 반복문 종료
    if len(number) == 1000:
        break



for i in range(0,1000):
    a = 'python scripts/stable_txt2img.py' + ' --seed ' + str(number.pop()) + ' --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 100 --ckpt ./logs/chim_input2022-11-26T22-54-09_chim2/checkpoints/epoch=000006.ckpt --prompt "a man of sks cartoon"'
    os.system(a)
'''


json_root = []
save_root = []
#save_path = "./output_LDY/"


save_path = 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230309\\'
#save_path = 'Z:\\PCL\\LDY\\output_front3d_6000\\'

#dir_path = "G:3D_Front/3D-FRONT/3D-FRONT/"
dir_path = "G:\\3D_Front\\3D-FRONT\\3D-FRONT\\"

count = 0


for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.json' in file:
            file_path = os.path.join(root, file)
            json_root.append(file_path)
            name = os.path.basename(file_path)
            name = name.rstrip(".json")

            #os.mkdir(save_path + name)
            save_root.append(save_path + name)
            
            count = count + 1
            if (count==1):
                break


'''
json_root = ['G:\\3D_Front\\3D-FRONT\\3D-FRONT\\00154c06-2ee2-408a-9664-b8fd74742897.json']
save_root = ['C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230227\\00154c06-2ee2-408a-9664-b8fd74742897\\']
'''

#print(save_root)
print("json files : ", len(json_root))




'''
#for i in range(0,len(json_root)):
for i in range(0, 3000):
    os.mkdir(save_root[i])
'''

for i in range(0,len(json_root)):
#for i in range(0,3000):


    os.mkdir(save_root[i])

    #cmd = "python -m blenderproc run .\\front3d.py " + "--front=" + str(json_root[i]) + " --future_folder=G:\\3D-FUTURE-model --front_3D_texture_path=G:\\3D-FRONT-texture\\3D-FRONT-texture\\00cc8b1d-b284-4108-a48f-a18c320a9d3a\\ --output_dir=./output_LDY/ " + str(save_root[i])
    cmd = "python -m blenderproc run ./front3d_230222.py " + "--front=" + str(json_root[i]) + " --future_folder=G:\\3D_Front\\3D-FUTURE-model --front_3D_texture_path=G:\\3D_Front\\3D-FRONT-texture\\3D-FRONT-texture\\00cc8b1d-b284-4108-a48f-a18c320a9d3a\\ --output_dir=" + str(save_root[i])

    print(i, " : ", len(json_root))
    print("\n",cmd,"\n")


    os.system(cmd)




'''
f = open("shell_script.sh", 'w')
for i in range(0,len(json_root)):
    #print(save_root[i]+'/campose_0000.npy')
    if (os.path.exists(save_root[i]+'/campose_0000.npy')) == False:
        cmd = "rmdir /s /q " + save_root[i]
        print(cmd)
        os.system(cmd)
    else:
        task = os.path.basename(json_root[i])
        task = task.rstrip(".json")
        #data = "./voxel.out " + task + "\n"
        data = "/home/edge/projects/voxel/bin/x64/Release/voxel.out " + task + " \\n" + "\n"
        f.write(data)
f.close()
'''
