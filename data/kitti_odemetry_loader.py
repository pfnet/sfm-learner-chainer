import numpy as np
from path import Path
import scipy.misc


class KittiOdemetryLoader(object):
    def __init__(self, dataset_dir, static_frames_file=None,
                 img_height=128, img_width=416, seq_length=3,
                 train_list=None, val_list=None):
        dir_path = Path(__file__).realpath().dirname()
        static_frames_file = Path(static_frames_file)
        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['02', '03']
        self.data_list = []
        self.parse_data_list(train_list, 'train')
        # self.parse_data_list(val_list, 'val')
        self.scenes = []
        self.scenes_info = {}
        self.collect_train_folders()
        self.collect_static_frames(static_frames_file)

    def parse_data_list(self, data_list, label):
        with open(data_list, 'r') as f:
            data_list = f.readlines()
        data_list = [[label] + d[:-1].split(" ") for d in data_list]
        self.data_list += data_list

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = {}
        for fr in frames:
            if fr == '\n':
                continue
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
            if drive not in self.static_frames.keys():
                self.static_frames[drive] = []
            self.static_frames[drive].append(curr_fid)

    def collect_train_folders(self):
        for data in self.data_list:
            drive_set = self.dataset_dir/data[1]/data[2]
            self.scenes.append(drive_set)
            self.scenes_info[str(drive_set.name)] = [data[0], data[3], data[4]]

    def collect_scenes(self, drive):
        train_scenes = []
        for c in self.cam_ids:
            oxts = sorted((drive/'oxts'/'data').files('*.txt'))
            scene_data = {'cid': c, 'dir': drive, 'speed': [], 'frame_id': [], 'rel_path': drive.name + '_' + c}
            for n, f in enumerate(oxts):
                metadata = np.genfromtxt(f)
                speed = metadata[8:11]
                scene_data['speed'].append(speed)
                scene_data['frame_id'].append('{:010d}'.format(n))
            res = self.load_image(scene_data, 0)
            if res is None:
                return []
            scene_data['intrinsics'] = self.get_intrinsics(scene_data, res[1], res[2])
            train_scenes.append(scene_data)
        return train_scenes

    def get_scene_imgs(self, scene_data, from_speed=False):
        drive = str(scene_data['dir'].name)
        scene_info = self.scenes_info[drive]
        start = int(scene_info[1])
        end = int(scene_info[2])
        for (i, frame_id) in enumerate(scene_data['frame_id']):
            if (drive not in self.static_frames.keys()) or \
                   (frame_id not in self.static_frames[drive]):
                if int(frame_id) >= start and int(frame_id) <= end:
                    # print(frame_id, start, end)
                    yield self.load_image(scene_data, i)[0], frame_id

    def get_intrinsics(self, scene_data, zoom_x, zoom_y):
        calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + scene_data['cid']], (3, 4))
        intrinsics = P_rect[:, :3]
        intrinsics[0] *= zoom_x
        intrinsics[1] *= zoom_y
        return intrinsics

    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'image_{}'.format(scene_data['cid'])/'data'/scene_data['frame_id'][tgt_idx]+'.png'
        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data
