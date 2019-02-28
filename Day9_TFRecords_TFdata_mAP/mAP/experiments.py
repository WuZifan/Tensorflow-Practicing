import glob
import os
import sys

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def error(msg):
    print(msg)
    sys.exit(0)

if __name__ == '__main__':
    ground_truth_files_list = glob.glob('ground-truth/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")

    ground_truth_files_list.sort()
    print(ground_truth_files_list)

    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list[:10]:
        '''
                从ground-truth\\2007_000027.txt
                拿到了：2007_000027
            '''
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # 看看对应的图片是否存在
        # check if there is a correspondent predicted objects file
        if not os.path.exists('predicted/' + file_id + ".txt"):
            error_msg = "Error. File not found: predicted/" + file_id + ".txt\n"
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
            error(error_msg)

        # 拿到文件中的每行
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        '''
            1、对于这个图中的每个gt，拿到对应的类别和位置
            2、
        '''
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            # if class_name in args.ignore:
            #     continue
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

                '''
                记录每个种类出现过几次
                '''
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1
                '''
                    1、already_seen_classes:对于每个图像，每个类别仅统计一次。
                        1.1 比如这个图中有5个C类，在这里也只统计一次。
                    2、是为了计算平均每个图片有几个类别？？
                '''
                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

    print(gt_counter_per_class)
    print(counter_images_per_class)