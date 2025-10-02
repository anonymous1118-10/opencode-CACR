import json
def extract_st_ed_subtitles(subtitle_path,id,starttime,end_time):
    with open(subtitle_path, 'r') as file:
        subtitles = json.load(file)
        # subtitles=sub_file
    extract_datas=[]
    data=subtitles[id] if id in subtitles.keys() else None
    if data is None:
        return None
    for subtitle in data:
        start = subtitle.get('start', 0)
        end = subtitle.get('end', float('inf'))
        if (starttime>=start and starttime<=end) or (end_time>=start and end_time<=end) or  (start>=starttime and end<=end_time):
            extract_datas.append(subtitle)
    return extract_datas

def test01():
    id='-7p4awI21FI'
    extract=extract_st_ed_subtitles('train_data_prepare/NLPCC_2023_CMIVQA_TRAIN_DEV/subtitle.json',id,160,176)
    print(extract)
    print(len(extract))


def main():
    test01()

if __name__ == "__main__":
    main()
