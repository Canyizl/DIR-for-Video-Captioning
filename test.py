
def convert_data_to_coco_scorer_format(reference):
    reference_json = {}
    non_ascii_count = 0
    with open(reference, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            try:
                sent.encode('ascii', 'ignore').decode('ascii')
            except UnicodeDecodeError:
                non_ascii_count += 1
                continue
            if vid in reference_json:
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
            else:
                reference_json[vid] = []
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
    if non_ascii_count:
        print("=" * 20 + "\n" + "non-ascii: " + str(non_ascii_count) + "\n" + "=" * 20)
    return reference_json
