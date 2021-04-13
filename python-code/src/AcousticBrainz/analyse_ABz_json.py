import json


def recursive_dfs(doc, trace: list):
    for key in list(doc.keys()):
        if isinstance(doc[key], dict):
            recursive_dfs(doc[key], trace + [key])
        elif isinstance(doc[key], list):
            # print(f'array:  {trace}: {key}')
            continue
        elif isinstance(doc[key], str):
            print(f'string item: parents:{trace};  key:{key}')

        else:
            # print(f'terminal: {key}')
            continue


if __name__ == '__main__':
    sample_json_location = "/run/media/sharwinbobde/SharwinThesis/mlhd-ab-features/acousticbrainz-mlhd-0123/00" \
                           "/00a0a60c-dc7a-470d-a256-325322a5c01a.json"
    fp = open(sample_json_location, 'r')
    doc = json.load(fp)
    recursive_dfs(doc, [])
