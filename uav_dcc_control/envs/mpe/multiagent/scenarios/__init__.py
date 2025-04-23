import imp
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    # print(pathname)
    # print(osp.dirname(__file__))
    # /data/guanyu/pycharm/dynamic-coverage-control/uav_dcc_control/envs/mpe/multiagent/scenarios/coverage.py
    return imp.load_source('', pathname)
