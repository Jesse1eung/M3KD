import re
import html
from urllib import parse
import requests
import argparse
import json
from tqdm import tqdm
from pygtrans import Translate

GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'

def translate(text, to_language="auto", text_language="auto"):

    text = parse.quote(text)
    url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
    response = requests.get(url)
    data = response.text
    expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
    result = re.findall(expr, data)
    if (len(result) == 0):
        return ""

    return html.unescape(result[0])




print(translate("about your situation", "zh-CN","en")) #英语转汉语
# sbu/cc/vg caption str; coco [cap1, cap2, cap3]

def trans(caps, client, start=0):
    batch_size = 512
    total_num = sum([len(c) for c in caps])
    print("total num:", total_num)
    count = total_num // batch_size
    last = total_num - count*batch_size
    last = 0
    temp = []
    end = start*batch_size + batch_size
    for i in tqdm(range(start, start+10000)):
        cap_list = caps[i*batch_size: i*batch_size+batch_size]
        # t = []
        # cap_list = [t.extend(cap_l) for cap_l in cap_lists]
        try:
            texts = client.translate(cap_list)
        except OSError:
            client = Translate()
            texts = client.translate(cap_list)
            end = i * batch_size
            # print(i)
            # with open(save_file, 'w') as fw:
            #     json.dump(temp, save_file)
            #     fw.write(i*128)
            # temp = []
            # return temp, end
            # temp = []
        except KeyboardInterrupt:
            client = Translate()
            texts = client.translate(cap_list)
            end = i*batch_size
        temp.extend([text.translatedText for text in texts])

    if last > 0:
        texts = client.translate(caps[count*batch_size:])
        temp.extend([text.translatedText for text in texts])
        end = total_num
    return temp, end

def read_cap(ann_path, start=0):
    anns = json.load(open(ann_path, "r"))
    client = Translate()
    caps = [ann['caption'] for ann in anns]
    # lens = [i for j in range(len(cap)) for i,cap in enumerate(caps)]
    lens = []
    for i, cap in enumerate(caps):
        lens.extend([i] * len(cap))
    zn_caps, end = trans(caps, client, start=start)
    print(zn_caps[0])
    # for i in range(start*128, end):
    for i in range(len(zn_caps)):
        id_ann = lens[i+start*128]
        if 'zn_caption' not in anns[id_ann]:
            anns[id_ann]['zn_caption'] = []
        else:

            anns[id_ann]['zn_caption'].append(zn_caps[i])

    # for i,ann in tqdm(enumerate(anns)):
    #     caps = ann['caption']
    #     if 'coco' in ann_path:
    #         temp = []
    #         for cap in caps:
    #             zn_cap = translate(cap, "zh-CN","en")
    #             temp.append(zn_cap)
    #         anns[i]['caption'] = temp
    #     else:
    #         zn_cap = translate(caps, "zh-CN","en")
    #         anns[i]['caption'] = zn_cap

    return anns

def write_path(file_name):
    save_path_name = file_name.replace('annotations', 'zh')

    return save_path_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--ann_path", required=True, help="path to configuration file.")
    parser.add_argument("--start_id", type=int, default=0, help="path to configuration file.")
    args = parser.parse_args()


    ann_file = args.ann_path
    # 读取json
    # anns = json.load(open(ann_path, "r"))
    # client = Translate()
    start = args.start_id
    # start = 0

    # total_num = len(anns)
    # count = total_num // 128
    # last = total_num - count*128
    # temp = []
    # for i in tqdm(range(start, count)):

    # 然后开始翻译 检测oserror,此时写入新的翻译
    anns = read_cap(ann_file, start=start)
    save_file = write_path(ann_file)

    with open(save_file, 'w') as fw:
        fw.write('[')
        for i,ann in enumerate(anns):
            if i != len(anns):
                fw.write(json.dumps(ann)+',\n')
            else:
                fw.write(json.dumps(ann)+']')
        # fw.write(']')
        # json.dump(anns, fw)
