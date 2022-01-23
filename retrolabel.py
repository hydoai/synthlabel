'''
Read YOLO-formatted object tracking outputs and smartly re-label detections to be more accurate.
'''
import argparse
from pathlib import Path

mock_input = {
    # simulates: a class 80 object that initially appears as class 0, but when it becomes bigger (closer to camera), it becomes class 80
    '1': [
        {'frame': 1,
        'track_id': 1,
        'area':0.01,
        'class_id': 0},
        {'frame': 2,
        'track_id': 1,
        'area': 0.02,
        'class_id': 0},
        {'frame': 3,
        'track_id': 1,
        'area': 0.03,
        'class_id': 0},
        {'frame': 4,
        'track_id': 1,
        'area': 0.09,
        'class_id': 80},
        {'frame': 5,
        'track_id': 1,
        'area': 0.12,
        'class_id': 80},
    ]
}
mock_output = {
    # it was actually class 80 all along
    '1': [
        {'frame': 1,
        'track_id': 1,
        'area':0.01,
        'class_id': 80},
        {'frame': 2,
        'track_id': 1,
        'area': 0.02,
        'class_id': 80},
        {'frame': 3,
        'track_id': 1,
        'area': 0.03,
        'class_id': 80},
        {'frame': 4,
        'track_id': 1,
        'area': 0.09,
        'class_id': 80},
        {'frame': 5,
        'track_id': 1,
        'area': 0.12,
        'class_id': 80},
    ]
}


def read_annotations(path):
    # build dictionary of all detections in this video
    detections = {}
    filename_stem = None
    for detection_txt in (path/'labels').glob('*.txt'):
        lines = []
        filename_stem = detection_txt.stem.split('_')[0]
        with open(detection_txt, 'r') as file:
            for line in file:
                line = line.strip().split(',')
                frame, track_id, x, y, w, h, confidence, class_id = line
                lines.append({
                    'frame': int(frame),
                    'track_id': int(track_id),
                    'x': float(x),
                    'y': float(y),
                    'w': float(w),
                    'h': float(h),
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'area': float(w)*float(h)

                })
        detections.update({int(frame): lines})
    return detections, filename_stem

def re_key_dict(d):
    # reorganize detections dictionary with 'track_id' as key
    detections_by_track_id = {}
    for frame, dets_in_frame in d.items():
        for det in dets_in_frame:
            track_id = det['track_id']
            if track_id not in detections_by_track_id:
                detections_by_track_id[track_id] = []
            detections_by_track_id[track_id].append(det)
    return detections_by_track_id

def vote_class_by_area(d):
    # to determine 'class_id', take a vote from detections with area greater than average
    # since we're using area, this will significantly skew the selection toward large detections
    for track_id, dets_in_track in d.items():
        dets_in_track.sort(key=lambda x: x['area'], reverse=True)

        smallest_area = dets_in_track[-1]['area']
        largest_area = dets_in_track[0]['area']
        mid_area = (smallest_area + largest_area)/2

        dets_with_suffrage = [det for det in dets_in_track if det['area'] >= mid_area]
        count_class_ids = {}
        for det in dets_with_suffrage:
            class_id = det['class_id']
            if class_id not in count_class_ids:
                count_class_ids[class_id] = 0
            count_class_ids[class_id] += 1
        max_count = max(count_class_ids.values())
        elected_class_id = [k for k, v in count_class_ids.items() if v == max_count][0]

        for det in dets_in_track:
            det['class_id'] = elected_class_id
    return d

def un_re_key_dict(d):
    # reorganize detections dictionary with 'frame' as key
    detections_by_frame = {}
    for track_id, dets_in_track in d.items():
        for det in dets_in_track:
            frame = det['frame']
            if frame not in detections_by_frame:
                detections_by_frame[frame] = []
            detections_by_frame[frame].append(det)
    return detections_by_frame

def write_dict_to_txt(d, path, filename_stem):
    # create 'voted_labels' directory if it doesn't exist
    if not (path/'voted_labels').exists():
        (path/'voted_labels').mkdir()

    for frame, dets_in_frame in d.items():
        with open(path/'voted_labels'/f'{filename_stem}_{frame}.txt', 'w') as file:
            for det in dets_in_frame:
                file.write(f'{det["frame"]},{det["track_id"]},{det["x"]},{det["y"]},{det["w"]},{det["h"]},{det["confidence"]},{det["class_id"]}\n')

def main(args):
    for subdir in Path(args.dir).glob('*'):
        detections, filename_stem = read_annotations(subdir)
        re_keyed_detections = re_key_dict(detections)
        voted_detections = vote_class_by_area(re_keyed_detections)
        un_re_keyed_detections = un_re_key_dict(voted_detections)
        if args.dry_run:
            print(un_re_keyed_detections)
        else:
            write_dict_to_txt(un_re_keyed_detections, subdir, filename_stem)

def test_main():
    detections = mock_input
    re_keyed_detections = re_key_dict(detections)
    voted_detections = vote_class_by_area(re_keyed_detections)
    un_re_keyed_detections = un_re_key_dict(voted_detections)
    import IPython; IPython.embed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read YOLO-formatted object tracking outputs and smartly re-label detections to be more accurate.\nUsage: python3 retrolabel.py --dir output")
    parser.add_argument('--dir', default='output', type=str, help="Path to where autolabel.py should read and re-write labels. This directory should should be the output of autolabel.py, a directory for each video input containing the following subdirectories: 'clean_images', 'annotated_images', 'labels'")
    parser.add_argument('--dry-run', default=False, action='store_true', help="Don't actually make any filesystem changes")

    args = parser.parse_args()
    #test_main()
    main(args)
