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
save_path = "D:\\my_test_uk\\"

# dir_path = "G:3D_Front/3D-FRONT/3D-FRONT/"
dir_path = "D:\\3D_Front\\3D-FRONT\\3D-FRONT\\"

count = 0

for (root, directories, files) in os.walk(dir_path):
    files.sort()
    for file in files:
        if '.json' in file:
            count = count + 1
            if count <200:
                continue
            file_path = os.path.join(root, file)
            json_root.append(file_path)
            # print(os.path.basename(file_path))
            name = os.path.basename(file_path)
            name = name.rstrip(".json")
            if os.path.isdir(save_path + name) == False:
                os.mkdir(save_path + name)
            save_root.append(save_path + name)
            # print(file_path)
            count = count + 1
            if (count == 500):
                break

print(save_root)

print("json files : ", len(json_root))

for i in range(0, count):
    # for i in range(0, 1):
    # cmd = "python -m blenderproc run .\\front3d.py " + "--front=" + str(json_root[i]) + " --future_folder=G:\\3D-FUTURE-model --front_3D_texture_path=G:\\3D-FRONT-texture\\3D-FRONT-texture\\00cc8b1d-b284-4108-a48f-a18c320a9d3a\\ --output_dir=./output_LDY/ " + str(save_root[i])
    cmd = "python -m blenderproc run ./front3d_auto_uk.py " + "--front=" + str(json_root[
                                                                           i]) + " --future_folder=D:\\3D_Front\\3D-FUTURE-model\\ --front_3D_texture_path=D:\\3D_Front\\3D-FRONT-texture\\3D-FRONT-texture\\ --output_dir=" + str(
        save_root[i]) + '\\'

    print(i, " : ", len(json_root))
    print(cmd)
    try:
        os.system(cmd)
    except:
        continue
