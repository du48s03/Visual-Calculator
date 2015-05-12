import sys
import os
includespath = os.path.abspath('../includes')
sys.path.insert(0,includespath)
import evala as e
if __name__ == '__main__':
    username = sys.argv[1]
    e.exp_points(username)
    e.exp_line(username)
    e.exp_circle(username)
