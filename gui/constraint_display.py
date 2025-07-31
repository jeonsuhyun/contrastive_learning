import os
import yaml
import numpy as np
import torch
import pickle
from functools import partial

import argparse

from PyQt5.QtWidgets import QApplication, QPushButton, QSlider, QVBoxLayout, QLabel, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt
from ljcmp.utils.generate_environment import generate_environment
from ljcmp.utils.model_utils import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-E', type=str, default='panda_dual_orientation', help='panda_orientation, panda_dual, panda_dual_orientation, panda_triple')

args = parser.parse_args()

constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)

constraint_model, _ = load_model(args.exp_name, model_info, 
                                            load_validity_model=False)
pc = constraint.planning_scene

z_dim = model_info['z_dim']
c_dim = model_info['c_dim']

c_lb = model_info['c_lb']
c_ub = model_info['c_ub']

c_lb = np.array(c_lb)  
c_ub = np.array(c_ub)

app = QApplication([])
window = QWidget()
layout = QVBoxLayout()

label = QLabel()
label.setText('Latent code')
layout.addWidget(label)

latent_slider_list = []
latent_label_value_list = []

condition_slider_list = []
condition_label_value_list = []

z = np.zeros(z_dim)
c = (c_ub + c_lb) / 2

def update_and_viz():
    set_constraint(c)
    constraint_model.set_condition(c)
    q = constraint_model.to_state(z)
    pc.display(q)
    # r = constraint.project(q)
    # if r is True:
    #     pc.display(q)

def update_latent_value(idx, val):
    latent_label_value_list[idx].setText('{:.2f}'.format(val/100))
    z[idx] = val/100
    update_and_viz()

def update_condition_value(idx, val):
    condition_label_value_list[idx].setText('{:.2f}'.format(val/100))
    c[idx] = val/100
    update_and_viz()

for i in range(z_dim):
    h_layout = QHBoxLayout()
    label = QLabel()
    label.setText('z_{}'.format(i))
    slider = QSlider()
    slider.setOrientation(1)
    slider.setRange(-200, 200)
    slider.setValue(0)
    label_value = QLabel()
    label_value.setText('0.00')
    # label_value.setLineWidth(5)

    slider.valueChanged.connect(partial(update_latent_value, i))

    latent_slider_list.append(slider)
    latent_label_value_list.append(label_value)

    h_layout.addWidget(label, 1)
    h_layout.addWidget(slider, 5)
    h_layout.addWidget(label_value, 2, alignment=Qt.AlignRight)
    layout.addLayout(h_layout)

    # slider.setTickPosition(0.05)
    # slider.setTickInterval(0.05)

label = QLabel()
label.setText('Condition')
layout.addWidget(label)

for i in range(c_dim):
    h_layout = QHBoxLayout()
    label = QLabel()
    label.setText('c_{}'.format(i))
    slider = QSlider()
    slider.setOrientation(1)
    slider.setRange(int(c_lb[i]*100), int(c_ub[i]*100))
    slider.setValue(int(c_ub[i]*100 + c_lb[i]*100) // 2)

    label_value = QLabel()
    label_value.setText('{:.2f}'.format((c_ub[i] + c_lb[i])/2))

    slider.valueChanged.connect(partial(update_condition_value, i))

    condition_slider_list.append(slider)
    condition_label_value_list.append(label_value)

    h_layout.addWidget(label, 1)
    h_layout.addWidget(slider, 5)
    h_layout.addWidget(label_value, 1, alignment=Qt.AlignRight)
    layout.addLayout(h_layout)
    # slider.setTickPosition(0.05)
    # slider.setTickInterval(0.05)
    # layout.addWidget(slider)

window.setLayout(layout)
window.show()
app.exec()
