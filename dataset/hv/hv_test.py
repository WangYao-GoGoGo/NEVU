import json
if __name__ == '__main__':
    with open("values.json", 'r') as file:
        datas = json.load(file)
    datas = datas["values"]
    hv_names = [data["name"] for data in datas]
    hv_names_str = str(hv_names)
    print(datas)