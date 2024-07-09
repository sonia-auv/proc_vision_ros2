import yaml
PARAM_DIR = '/home/sonia/ssd/ros2_sonia_ws/src/proc_vision_ros2/proc_vision_ros2/'
PARAM_YAML = 'items.yaml'

class ItemsRobosub():
    def __init__(self):
        self.args = yaml.full_load(open(PARAM_DIR+PARAM_YAML))
        self.constante = self.args['constante_camera']

    def get_dist(self, h, cls_name, target_index=1):
        print(cls_name)
        try:
            if cls_name == 'torpille_target':
                dist = self.get_dist_target(h, target_index)
            else:
                dist = self.args[cls_name]['h'] / (h * self.constante)
                return dist
        except:
            return -1
        
    def get_dist_target(self, h, index_target):
        try:
            dist = self.args['torpille_target'][index_target]['h'] / (h * self.constante)
            return dist
        except:
            return -1

    def get_target_index(self, h, dist):
        try:
            measured_h = dist * self.constante * h
            best_error = 1000
            best_target = 1
            for i, size in self.args['torpille_target'].items():
                error = size['h'] - measured_h
                print(error, best_error)
                if abs(error) < abs(best_error):
                    best_error = error
                    best_target = i
            return best_target
        except:
            return -1


def main():
    items = ItemsRobosub()
    dist = items.get_dist(200, 'torpille')
    print(dist, items.get_target_index(50, dist))

if __name__ == '__main__':
    main()
