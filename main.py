"""
PyTorch Implementation of ID-based Graph Recommender Systems
Author: Yi Zhang (zhangyi.ahu@gmail.com)
"""
__author__ = "Yi Zhang"

import torch
import Parser
import utility.utility_data.data_loader as data_loader
import utility.utility_function.tools as tools
import os
import logging

print('ID-GRec: PyTorch Implementation of ID-based Graph Recommender Systems')
print('-' * 100)

print('Step 1: General parameter setting reading...')
print('-' * 100)
args = Parser.parse_args()

if args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

if args.seed_flag:
    tools.set_seed(args.seed)

print('Step 2: Select model...')
print('\t 1. MFBPR    \t 2. GCMC    \t 3. GCCF      \t 4. NGCF      \t 5. LightGCN')
print('\t 6. IMPGCN   \t 7. SGL     \t 8. CVGA      \t 9. SimGCL    \t 10.XSimGCL')
print('\t 11.DirectAU \t 12.NCL     \t 13.HCCF      \t 14.LightGCL  \t 15.VGCL*')
print('\t 16.DCCF     \t 17.CGCL    \t 18.GraphAU*  \t 19.MAWU      \t 20.RecDCL')
print('\t 21.BIGCF    \t 22.SCCF    \t 23.EGCF      \t 24.LightGODE \t 25.LightGCN_pp')
print('\t 26.MixRec')
print('\t Models marked with * are still being tested, so stay tuned.')
print('-' * 100)

model_list = {"0": "unknown", "1": "MFBPR", "2": "GCMC", "3": "GCCF", "4": "NGCF", "5": "LightGCN",
              "6": "IMPGCN", "7": "SGL", "8": "CVGA", "9": "SimGCL", "10": "XSimGCL", "11": 'DirectAU',
              "12": "NCL", "13": "HCCF", "14": "LightGCL", "16": "DCCF", "17": "CGCL", "19": "MAWU",
              "20": "RecDCL", "21": "BIGCF", "22": "SCCF", "23": 'EGCF', "24": "LightGODE",
              "25": "LightGCN_pp", "26": "MixRec",
              }

if args.model == "unknown":
    while True:
        selected_num = input('Please input the identifier of the model:')
        if selected_num not in model_list.keys() or selected_num == '0':
            print("Input Error. Please select from the list of implemented models and try again.")
        else:
            break
else:
    selected_num = "0"
    model_list[selected_num] = args.model


print('Step 3.1: Loading configuration file...')

import_str = 'from models.' + model_list[selected_num] + " import Trainer"
config_str = './configure/' + model_list[selected_num] + ".txt"
exec(import_str)

config = tools.read_configuration(config_str, model_list[selected_num])

if not os.path.exists('log/' + model_list[selected_num]):
    os.mkdir('log/' + model_list[selected_num])
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
logfile = logging.FileHandler('log/{}/{}.log'.format(model_list[selected_num], config['dataset']), 'a', encoding='utf-8')
logfile.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
logfile.setFormatter(formatter)
logger.addHandler(logfile)

print('Step 3.2: Loading dataset file...')

dataset = data_loader.Data(config['dataset_path'] + config['dataset'], config)

logger.info("Run with " + model_list[selected_num] + " on " + config['dataset'])
logger.info(dataset.get_statistics())

print('-' * 100)
print('\t Step 3.3: Init the Recommendation Model:')

recommener = None
model_str = 'recommener = Trainer(args, config, dataset, device, logger)'

exec(model_str)

print('\t model: ', model_list[selected_num])

for key in config:
    print("\t " + str(key) + " : " + str(config[key]))
    logger.info(str(key) + " : " + str(config[key]))

print('-' * 100)
print("Step 4: Model training and testing process:")

recommener.train()
